# Gemini Integration for Project Helios

## Overview

This document describes the Gemini AI integration for the blind user camera assistant. The goal is to provide real-time audio narration based on YOLO object detection + camera images.

## Current Status: 🔴 In Progress

The Gemini Live API integration is **not yet working**. We've set up the infrastructure but are encountering issues with the Vertex AI Live API connection.

## What We Built

### Files Created

- **`gemini_service.py`** - Main service class for Gemini integration
  - `GeminiLiveNarrator` - Async class using Vertex AI Live API
  - `GeminiNarrator` - Sync wrapper
  - `NarrationResult` - Dataclass with audio, transcript, and timing metrics
  - Caching layer (TTL-based) for repeated scenes

- **`test_gemini_service.py`** - Test script with mock YOLO data
  - Tests async narrator, streaming, and sync wrapper
  - Saves output as WAV files for playback

### Dependencies Added to requirements.txt

```
google-genai>=1.0.0
python-dotenv>=1.0.0
cachetools>=5.0.0
```

### Environment Variables (.env)

```
GEMINI_API_KEY=<your-api-key>                    # For basic Gemini API (works)
GOOGLE_APPLICATION_CREDENTIALS=./vertex-api-key.json  # For Vertex AI
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

## Architecture

```
YOLO Server (friend's piece)
    ↓ scene_analysis JSON + base64 image
GeminiLiveNarrator
    ├─ Cache check (hash of summary + labels + image)
    │   └─ HIT: Return cached audio (<1ms)
    └─ MISS:
        ├─ Connect to Vertex AI Live API (WebSocket)
        ├─ Send image + YOLO summary
        ├─ Receive audio stream (24kHz PCM)
        ├─ Cache result
        └─ Return NarrationResult
```

## What Works ✅

1. **Basic Gemini API** (`gemini_test.py`)
   - Text generation with `gemini-2.0-flash`
   - Streaming responses
   - Uses simple API key authentication

2. **Service Structure**
   - Caching mechanism (TTL-based, semantic keys)
   - Image decoding (base64 with data URL prefix handling)
   - Prompt building from YOLO scene_analysis
   - WAV file export

## What's Broken ❌

1. **Vertex AI Live API Connection**
   - Model names keep changing/not being found
   - Tried: `gemini-2.0-flash-exp`, `gemini-2.0-flash-live-001`, `gemini-live-2.5-flash-native-audio`
   - Error: WebSocket connection closes with policy violation or model not found

2. **Native Audio Model**
   - `gemini-live-2.5-flash-native-audio` only outputs audio (no text)
   - Need to configure `response_modalities=["AUDIO"]` correctly

## APIs Comparison

| API | Auth | Live API | Latency | Setup |
|-----|------|----------|---------|-------|
| Google AI Studio | API Key | ❌ No | ~500ms | Easy |
| Vertex AI | Service Account | ✅ Yes | ~100-300ms | Complex |

We chose Vertex AI for the Live API's lower latency, but setup is more complex.

## Next Steps

### Immediate (to get it working)

1. **Verify Vertex AI API is enabled**
   - Check Google Cloud Console → APIs & Services
   - Ensure "Vertex AI API" is enabled (not just "Generative Language API")

2. **Check service account permissions**
   - Service account needs "Vertex AI User" role
   - Verify JSON key is valid: `cat vertex-api-key.json`

3. **Try different model names**
   - Check available models: `client.models.list()`
   - Look for models with "live" in the name

4. **Fallback: Use regular streaming API**
   - If Live API doesn't work, use `generate_content_stream` with images
   - Still fast (~500ms), works with basic API key
   - Generate text, then use separate TTS (Google Cloud TTS or browser TTS)

### Future Enhancements

1. **Persistent WebSocket session**
   - Keep connection open for continuous video stream
   - Reduces per-frame connection overhead

2. **Spatial Context Manager**
   - Track user movement (turns, distance)
   - Maintain scene memory
   - Transform object positions based on movement
   - Example: "The door you passed is now behind you on the left"

3. **Smart narration throttling**
   - Don't narrate every frame
   - Only narrate on significant scene changes
   - Emergency (vehicle nearby) = immediate narration

4. **Audio streaming to client**
   - Stream audio chunks via Socket.IO as they arrive
   - Client starts playback before full response

## Testing

```bash
# Basic Gemini test (works)
python gemini_test.py

# Full service test (currently broken)
python test_gemini_service.py

# With custom image
python test_gemini_service.py /path/to/image.jpg
```

## Resources

- [Gemini Live API Docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
- [Vertex AI vs AI Studio](https://ai.google.dev/gemini-api/docs/migrate-to-cloud)
- [google-genai Python SDK](https://github.com/google/generative-ai-python)

## Notes from Development Session

- The `google.generativeai` package is deprecated → use `google.genai`
- Live API requires Vertex AI, not the basic Generative Language API
- Vertex AI uses OAuth/service accounts, not simple API keys
- Model naming is inconsistent between docs and actual API
- Native audio models only output audio, not text (need transcription separately)
