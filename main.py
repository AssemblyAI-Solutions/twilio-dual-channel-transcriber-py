from fastapi import FastAPI, WebSocket, Response, BackgroundTasks
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Start
from twilio.rest import Client
import uvicorn
from pyngrok import ngrok
import os
import asyncio
import pyaudio
import audioop
import base64
import json
import logging
from assemblyai_transcriber import run_transcription

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twilio credentials
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
twilio_number = os.environ['TWILIO_PHONE_NUMBER']
assemblyai_api_key = os.environ['ASSEMBLYAI_API_KEY']

client = Client(account_sid, auth_token)

# PyAudio setup
CHUNK = 160  # 20ms at 8kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000  # Twilio expects 8kHz

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

inbound_queue = asyncio.Queue()
outbound_queue = asyncio.Queue()
transcription_task = None

@app.get("/")
async def root():
    return {"message": "WebSocket server is running"}

async def start_transcription():
    global transcription_task
    if transcription_task is None or transcription_task.done():
        transcription_task = asyncio.create_task(run_transcription(assemblyai_api_key, inbound_queue, outbound_queue))

@app.post("/start-transcription")
async def start_transcription_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_transcription)
    return {"message": "Transcription started"}

@app.websocket("/ws/interact")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    try:
        while True:
            try:
                # Read audio from microphone
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Convert to μ-law
                mu_law_data = audioop.lin2ulaw(data, 2)  # 2 bytes per sample for paInt16
                # Encode to base64
                encoded_data = base64.b64encode(mu_law_data).decode('utf-8')
                
                # Receive data from Twilio
                received_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                received_json = json.loads(received_data)
                
                if received_json['event'] == 'start':
                    stream_sid = received_json['start']['streamSid']
                    logger.info(f"Stream started with SID: {stream_sid}")
                elif received_json['event'] == 'media':
                    # Decode from base64
                    decoded_data = base64.b64decode(received_json['media']['payload'])
                    # Convert from μ-law to linear
                    linear_data = audioop.ulaw2lin(decoded_data, 2)
                    # Play received audio
                    stream.write(linear_data)
                
                # Send audio to Twilio
                if stream_sid:
                    payload = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": encoded_data
                        }
                    }
                    await websocket.send_text(json.dumps(payload))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/media")
async def media_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Media logger connection accepted")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['event'] == "connected":
                logger.info("Connected Message received: %s", message)
            elif data['event'] == "start":
                logger.info("Start Message received: %s", message)
            elif data['event'] == "media":
                track = data['media'].get('track', 'unknown')
                # Decode from base64
                decoded_data = base64.b64decode(data['media']['payload'])
                if track == 'inbound':
                    # logger.info(f"Received inbound audio data, length: {len(decoded_data)}")
                    await inbound_queue.put(decoded_data)
                elif track == 'outbound':
                    # logger.info(f"Received outbound audio data, length: {len(decoded_data)}")
                    await outbound_queue.put(decoded_data)
                else:
                    logger.warning(f"Received audio data with unknown track: {track}")
            elif data['event'] == "closed":
                logger.info("Closed Message received: %s", message)
                break
    except Exception as e:
        logger.error("Error in Media WebSocket: %s", str(e))
    finally:
        logger.info("Media logger connection closed.")
        await inbound_queue.put(None)
        await outbound_queue.put(None)

@app.post("/make_call")
async def make_call(phone_number: str, background_tasks: BackgroundTasks):
    logger.info(f"Making call to {phone_number}")
    logger.info(f"Ngrok URL: {ngrok_url}")
    try:
        call = client.calls.create(
            url=f'https://{ngrok_url}/outbound-call',
            to=phone_number,
            from_=twilio_number
        )
        # Start transcription when the call is initiated
        background_tasks.add_task(start_transcription)
        return {"message": "Call initiated", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Error making call: {e}")
        return {"message": "Error making call", "error": str(e)}

@app.post("/outbound-call")
async def outbound_call(background_tasks: BackgroundTasks):
    response = VoiceResponse()
    
    # Start unidirectional stream with both tracks
    start = Start()
    start.stream(url=f'wss://{ngrok_url}/ws/media', track='both_tracks')
    response.append(start)

    # Start bidirectional stream
    connect = Connect()
    connect.stream(url=f'wss://{ngrok_url}/ws/interact')
    response.append(connect)
    
    # Start transcription when the call is connected
    background_tasks.add_task(start_transcription)
    
    # Create a FastAPI Response with the correct Content-Type
    return Response(content=str(response), media_type="application/xml")

if __name__ == "__main__":
    # Start ngrok
    ngrok_tunnel = ngrok.connect(8000, bind_tls=True)
    ngrok_url = ngrok_tunnel.public_url.replace("https://", "")
    
    logger.info(f"Ngrok URL: {ngrok_url}")
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)