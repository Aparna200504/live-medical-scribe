import asyncio
import websockets

async def test():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send("stop")
        response = await websocket.recv()
        print(response)

asyncio.run(test())