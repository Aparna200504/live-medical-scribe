import json

import pytest

from llm import _parse_summary


VALID_SUMMARY = {
    "patient_details": {
        "name": "Alex Smith",
        "age": "42",
        "sex": "Not mentioned",
        "identifiers": [],
    },
    "chief_complaint": "Headache",
    "history_present_illness": "Not mentioned",
    "symptoms": {"positive": ["Headache"], "negative": ["No fever"]},
    "past_medical_history": "Not mentioned",
    "medication_history": "Not mentioned",
    "clinical_observations": "Not mentioned",
    "assessment": "Not mentioned",
    "plan": "Not mentioned",
}


def test_clinical_summary_schema_accepts_required_fields():
    summary = _parse_summary(json.dumps(VALID_SUMMARY))

    assert summary.patient_details.name == "Alex Smith"
    assert summary.symptoms.negative == ["No fever"]


def test_clinical_summary_schema_rejects_missing_fields():
    invalid_summary = dict(VALID_SUMMARY)
    invalid_summary.pop("plan")

    with pytest.raises(RuntimeError):
        _parse_summary(json.dumps(invalid_summary))
