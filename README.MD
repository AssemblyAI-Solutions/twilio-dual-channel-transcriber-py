# Twilio Outbound Call Transcription POC

This Proof of Concept (POC) demonstrates how to perform an outbound Twilio call between a computer's audio (laptop microphone and speaker) and a phone number, while using AssemblyAI's streaming API and WebSockets to create a live transcript of the Twilio conversation. The project establishes two separate WebSocket connections for transcribing each audio track (inbound and outbound).

## Features

- Outbound Twilio calls using computer audio
- Real-time transcription using AssemblyAI's streaming API
- Separate transcription for inbound and outbound audio tracks
- Live display of transcripts with timestamps
- Automatic ngrok tunnel setup using pyngrok for exposing local server to the internet

## Transcript Format

The transcripts are displayed in the following format:

```
<Track>: <Transcript Text> [<Start Timestamp> - <End Timestamp>]
```

Example:
```
Inbound: Hey, what's going on? [0 - 1500]
Outbound: Hello. [1000 - 1800]
```

Timestamps are in milliseconds relative to the start of the audio stream for each track.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/twilio-outbound-call-transcription-poc.git
   cd twilio-outbound-call-transcription-poc
   ```

2. Create and activate a virtual environment, then install the required dependencies:
   ```
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS and Linux:
   source venv/bin/activate

   # Install the required dependencies
   pip install -r requirements.txt
   ```

3. Create a .env file with your environment variables:
   ```
   export TWILIO_ACCOUNT_SID='your_account_sid'
   export TWILIO_AUTH_TOKEN='your_auth_token'
   export TWILIO_PHONE_NUMBER='your_twilio_phone_number'
   export ASSEMBLYAI_API_KEY='your_assemblyai_api_key'
   ```

4. Update the phone number in `make_call.py`:
   Open `make_call.py` and replace the phone number in the `params` dictionary with the desired outbound call number.

5. Run the main application:
   ```
   python main.py
   ```
   This will start the FastAPI server and automatically set up an ngrok tunnel using pyngrok.

6. In a separate terminal, run the make_call script to initiate an outbound call:
   ```
   python make_call.py
   ```

## Technical Details

- This project calls the AssemblyAI WebSocket directly instead of using the AssemblyAI Python SDK. This is because the SDK's async context is based on threading and not asyncio, and asyncio is more efficient for this use case. Future updates to the AssemblyAI SDK are planned to support asyncio-based async operations.

- The Twilio code sets up two media streams:
  1. A unidirectional async stream used for transcription
  2. A bidirectional stream used for streaming laptop input and output audio

  The unidirectional stream is crucial for transcription and is the most important part when adapting this setup for custom use cases.

- ngrok is used to expose the local server to the internet, allowing Twilio to communicate with your application. The ngrok tunnel is automatically set up using pyngrok when you run `main.py`. The ngrok URL is printed to the console and used for configuring the Twilio webhook URLs.

## Ngrok and Pyngrok

This project uses [ngrok](https://ngrok.com/) via the [pyngrok](https://pyngrok.readthedocs.io/) library to create a secure tunnel to your localhost. This allows Twilio to send webhook requests to your local development environment.

When you run `main.py`, pyngrok automatically starts an ngrok tunnel and provides a public URL. This URL is then used to configure the Twilio webhook URLs for the call.

Note: If you're running this application frequently, you might want to consider [signing up for a free ngrok account](https://dashboard.ngrok.com/signup) and using an auth token to avoid rate limits.

## Limitations and Future Work

- Currently, this POC only supports outbound calls. Future iterations aim to include functionality for transcribing inbound calls as well.

## Contributing

Contributions to improve the POC or extend its functionality are welcome. Please feel free to submit issues or pull requests.