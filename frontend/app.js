let socket = null;
let audioContext = null;
let mediaStream = null;
let processor = null;
let source = null;
let mutedOutput = null;
let recording = false;
let fullTranscript = '';
let chunkCount = 0;
let summaryData = null;

const $ = id => document.getElementById(id);
const startBtn = $('startBtn');
const stopBtn = $('stopBtn');
const clearBtn = $('clearBtn');
const saveBtn = $('saveBtn');
const status = $('status');
const transcriptBox = $('transcript');
const message = $('message');
const summaryBox = $('summary');

startBtn.onclick = startRecording;
stopBtn.onclick = stopRecording;
clearBtn.onclick = clearAll;
saveBtn.onclick = saveTranscript;
$('clearSummaryBtn').onclick = clearSummary;
$('copyTranscriptBtn')?.addEventListener('click', () => copyText(fullTranscript, 'Transcript copied'));
$('copySummaryBtn').onclick = () => copyText(formatSummary(summaryData), 'Clinical summary copied');

function wsUrl() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  // Frontend runs on :3000 during local development; backend runs on :8000.
  const host = location.hostname || 'localhost';
  const port = (host === 'localhost' || host === '127.0.0.1') ? '8000' : location.port;
  return `${protocol}://${host}:${port}/ws`;
}

async function startRecording() {
  clearError();
  fullTranscript = '';
  chunkCount = 0;
  summaryData = null;
  renderTranscript();
  clearSummary();
  setStatus('Connecting…', 'busy');
  setMessage('Connecting to FastAPI and preparing VAD + Whisper…');
  startBtn.disabled = true;

  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone access is not available in this browser.');
    socket = new WebSocket(wsUrl());
    socket.onmessage = handleMessage;
    socket.onclose = () => { cleanupAudio(); if (status.className.includes('recording')) setStatus('Disconnected','idle'); };
    await waitForSocket(socket);

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    await audioContext.resume();
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    mutedOutput = audioContext.createGain();
    mutedOutput.gain.value = 0;
    processor.onaudioprocess = sendAudio;
    source.connect(processor);
    processor.connect(mutedOutput);
    mutedOutput.connect(audioContext.destination);
    recording = true;
    stopBtn.disabled = false;
    setStatus('Listening', 'recording');
    setPipeline('capture');
    setMessage('Listening — VAD will remove silence and background noise before ASR.');
  } catch (err) {
    cleanupAudio();
    if (socket) socket.close();
    socket = null;
    startBtn.disabled = false;
    setStatus('Ready','idle');
    showError(err.name === 'NotAllowedError' ? 'Microphone permission was denied.' : err.message);
  }
}

function sendAudio(event) {
  if (!recording || !socket || socket.readyState !== WebSocket.OPEN) return;
  const input = event.inputBuffer.getChannelData(0);
  const ratio = 16000 / audioContext.sampleRate;
  const outputLength = Math.floor(input.length * ratio);
  const pcm = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const sample = input[Math.min(input.length - 1, Math.floor(i / ratio))];
    const clipped = Math.max(-1, Math.min(1, sample));
    pcm[i] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
  }
  socket.send(pcm.buffer);
}

function stopRecording() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  recording = false;
  stopBtn.disabled = true;
  setStatus('Analyzing…', 'busy');
  setPipeline('llm');
  setMessage('Recording stopped. Finishing speech recognition, then sending the full transcript to the LLM…');
  cleanupAudio();
  socket.send('stop');
}

function handleMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === 'status') {
    setStatus(msg.message, msg.stage === 'recording' ? 'recording' : 'busy');
    const map = { initializing:'vad', transcribing:'asr', summarizing:'llm', recording:'capture', ready:'vad' };
    if (map[msg.stage]) setPipeline(map[msg.stage]);
    if (msg.stage === 'recording') $('vadState').textContent = 'Speech detected';
    else if (msg.stage === 'transcribing') $('vadState').textContent = 'Speech segment';
  }
  if (msg.type === 'transcript') {
    fullTranscript = `${fullTranscript} ${msg.text}`.trim();
    chunkCount += 1;
    renderTranscript();
    $('vadState').textContent = 'Active';
    setMessage('Transcript updated live — VAD speech segment sent to Whisper.');
  }
  if (msg.type === 'summary') {
    summaryData = msg.data;
    if (msg.data?.error) showError(msg.data.error);
    renderSummary(summaryData);
    setStatus('Complete', 'done');
    setPipeline('llm');
    startBtn.disabled = false;
    $('vadState').textContent = 'Complete';
    setMessage('Session complete. Review the structured clinical summary before use.');
  }
  if (msg.type === 'error') {
    showError(msg.message);
    setStatus('Error','error');
    startBtn.disabled = false;
  }
}

function renderTranscript() {
  transcriptBox.innerHTML = fullTranscript ? escapeHtml(fullTranscript) : '<span class="placeholder">Transcribed speech will appear here...</span>';
  $('wordCount').textContent = fullTranscript ? fullTranscript.split(/\s+/).length : 0;
  $('chunkCount').textContent = chunkCount;
  transcriptBox.scrollTop = transcriptBox.scrollHeight;
}

function renderSummary(s) {
  if (!s || s.error) return;
  const p = s.patient_details || {};
  $('patientName').textContent = p.name || 'Not mentioned';
  $('patientAge').textContent = p.age || 'Not mentioned';
  $('patientSex').textContent = p.sex || 'Not mentioned';

  const sections = [
    ['Patient details', `<p><b>Name:</b> ${esc(p.name || 'Not mentioned')} &nbsp; <b>Age:</b> ${esc(p.age || 'Not mentioned')} &nbsp; <b>Sex:</b> ${esc(p.sex || 'Not mentioned')}</p><p class="patient-identifiers"><b>Identifiers:</b> ${esc((p.identifiers || []).join(', ') || 'Not mentioned')}</p>`],
    ['Chief complaint', `<p>${esc(s.chief_complaint)}</p>`],
    ['History of present illness', `<p>${esc(s.history_present_illness)}</p>`],
    ['Symptoms — positive', list(s.symptoms?.positive)],
    ['Symptoms — stated negatives', list(s.symptoms?.negative)],
    ['Past medical history', `<p>${esc(s.past_medical_history)}</p>`],
    ['Medication history', `<p>${esc(s.medication_history)}</p>`],
    ['Clinical observations', `<p>${esc(s.clinical_observations)}</p>`],
    ['Assessment', `<p>${esc(s.assessment)}</p>`],
    ['Plan', `<p>${esc(s.plan)}</p>`]
  ];
  summaryBox.innerHTML = sections.map(([title, content]) => `<section class="summary-section"><h3>${title}</h3>${content}</section>`).join('');
}

function list(items) { return items?.length ? `<ul>${items.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '<p>None stated</p>'; }
function esc(v) { return String(v ?? 'Not mentioned').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function escapeHtml(v) { return esc(v); }

function formatSummary(s) {
  if (!s || s.error) return s?.error || '';
  const p = s.patient_details || {};
  return `PATIENT DETAILS\nName: ${p.name || 'Not mentioned'}\nAge: ${p.age || 'Not mentioned'}\nSex: ${p.sex || 'Not mentioned'}\nIdentifiers: ${(p.identifiers||[]).join(', ') || 'Not mentioned'}\n\nCHIEF COMPLAINT\n${s.chief_complaint}\n\nHISTORY OF PRESENT ILLNESS\n${s.history_present_illness}\n\nSYMPTOMS\nPositive: ${(s.symptoms?.positive||[]).join(', ') || 'None stated'}\nNegative: ${(s.symptoms?.negative||[]).join(', ') || 'None stated'}\n\nPAST MEDICAL HISTORY\n${s.past_medical_history}\n\nMEDICATION HISTORY\n${s.medication_history}\n\nCLINICAL OBSERVATIONS\n${s.clinical_observations}\n\nASSESSMENT\n${s.assessment}\n\nPLAN\n${s.plan}`;
}

function clearAll() {
  recording = false;
  cleanupAudio();
  if (socket) socket.close();
  socket = null;
  fullTranscript = ''; chunkCount = 0; summaryData = null;
  renderTranscript(); clearSummary(); clearError();
  $('vadState').textContent = 'Idle';
  setStatus('Ready','idle'); setPipeline(null);
  setMessage('Ready — click Start Mic and speak naturally.');
  startBtn.disabled = false;
}
function clearSummary() {
  summaryBox.innerHTML = '<div class="empty-summary"><div class="empty-icon">✦</div><strong>Clinical summary will appear here</strong><span>Stop the recording to send the complete transcript to the LLM.</span></div>';
  $('patientName').textContent='Not mentioned'; $('patientAge').textContent='Not mentioned'; $('patientSex').textContent='Not mentioned';
}
function saveTranscript() {
  if (!fullTranscript) return alert('No transcript to save yet.');
  const blob = new Blob([fullTranscript], {type:'text/plain'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download='medical-transcript.txt'; a.click(); URL.revokeObjectURL(a.href);
}
async function copyText(text, messageText) { if (!text) return; try { await navigator.clipboard.writeText(text); setMessage(messageText); } catch { showError('Clipboard access is unavailable.'); } }
function setStatus(text,state){ status.textContent=text; status.className=`status ${state}`; }
function setPipeline(stage){ document.querySelectorAll('.pipeline-step').forEach(x=>x.classList.toggle('active',x.dataset.stage===stage)); }
function setMessage(text){ message.className='message'; message.innerHTML=esc(text); }
function showError(text){ message.className='message error'; message.textContent=text; }
function clearError(){ if(message.classList.contains('error')) setMessage('Ready — click Start Mic and speak naturally.'); }
function cleanupAudio(){ if(processor) processor.disconnect(); if(source) source.disconnect(); if(mutedOutput) mutedOutput.disconnect(); if(mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); if(audioContext) audioContext.close(); processor=source=mutedOutput=mediaStream=audioContext=null; recording=false; stopBtn.disabled=true; }
function waitForSocket(ws){ return new Promise((resolve,reject)=>{ ws.onopen=resolve; ws.onerror=()=>reject(new Error('Could not connect to FastAPI. Start the backend on port 8000.')); }); }
