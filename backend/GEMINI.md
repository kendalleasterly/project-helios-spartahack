# Gemini Integration for Project Helios

## Overview

This document describes the Gemini AI integration for the blind user camera assistant. The goal is to provide real-time audio narration based on YOLO object detection + camera images.

## Current Status: 🟢 Working

The Gemini Live API integration is **working**. Audio narration is generated via Vertex AI Live API with smart throttling to avoid excessive narration.

## What We Built

### Files Created

- **`gemini_service.py`** - Main service classes for Gemini integration
  - `GeminiLiveNarrator` - Async class using Vertex AI Live API (one connection per request)
  - `GeminiLiveSession` - **NEW**: Persistent session with continuous context updates
  - `GeminiNarrator` - Sync wrapper (handles nested event loops)
  - `NarrationResult` - Dataclass with audio, transcript, and timing metrics
  - `RateLimiter` - Sliding window rate limiter to prevent API abuse
  - Caching layer (TTL-based) for repeated scenes
  - Semantic similarity for smart scene change detection

- **`test_gemini_service.py`** - Test script with mock YOLO data
  - Tests async narrator, streaming, and sync wrapper
  - Saves output as WAV files for playback

### Dependencies Added to requirements.txt

```
google-genai>=1.0.0
python-dotenv>=1.0.0
cachetools>=5.0.0
sounddevice>=0.4.0  # For server-side audio playback
```

### Environment Variables (.env)

```
GOOGLE_APPLICATION_CREDENTIALS=./vertex-api-key.json  # For Vertex AI
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

## Architecture

### Per-Request Mode (GeminiLiveNarrator)

```
YOLO Server
    ↓ scene_analysis JSON + base64 image
GeminiLiveNarrator
    ├─ Rate limit check (10 calls/min default)
    ├─ Cache check (hash of summary + labels + image)
    │   └─ HIT: Return cached audio (<1ms)
    └─ MISS:
        ├─ Connect to Vertex AI Live API (WebSocket)
        ├─ Send image + YOLO summary
        ├─ Receive audio stream (24kHz PCM)
        ├─ Cache result
        └─ Return NarrationResult
```

### Persistent Session Mode (GeminiLiveSession) - RECOMMENDED

```
Server Startup
    ↓
GeminiLiveSession.connect()
    ↓ (persistent WebSocket)

Frame 1 → update_context() → [no response, just feeds context]
Frame 2 → update_context() → [no response]
Frame 3 → update_context() → [no response]
Frame 4 → process_frame() → similarity < 0.7 → request_narration() → Audio!
Frame 5 → update_context() → [no response]
...
Frame N → EMERGENCY → request_narration() → Immediate Audio!
...
User asks "What's ahead?" → answer_question() → Audio response
```

## Classes & API

### GeminiLiveNarrator (Simple Mode)
```python
narrator = GeminiLiveNarrator(
    model="gemini-live-2.5-flash-native-audio",
    cache_ttl=30,
    rate_limit_calls=10,
    rate_limit_window=60.0
)
result = await narrator.narrate_with_image(scene_analysis, image_base64)
```

### GeminiLiveSession (Smart Mode)
```python
session = GeminiLiveSession(
    min_narration_interval=5.0,  # Seconds between auto-narrations
    rate_limit_calls=20,
    rate_limit_window=60.0
)
await session.connect()

# Main loop - call this for every frame
result = await session.process_frame(scene_analysis, image_base64)
if result:
    play_audio(result.audio_data)

# For user questions (when phone audio is ready)
result = await session.answer_question(
    question="What's in front of me?",
    scene_analysis=current_scene,
    image_base64=current_image
)
```

### Smart Narration Logic

| Trigger | Action |
|---------|--------|
| Emergency (vehicle close) | Immediate narration |
| New object type appeared | Narrate |
| Object moved far → close | Narrate |
| Similarity < 0.7 + time > 5s | Narrate |
| Similarity >= 0.7 | Skip (same scene) |
| Time < 5s since last | Skip (throttled) |

### Semantic Similarity (`_scene_similarity`)

Instead of exact hash matching (which fails with real camera footage), we use semantic comparison:

- Count objects by label (person: 2, chair: 3)
- Count by (label, distance) tuples
- Detect new object types → definitely narrate
- Detect distance changes (far→close) → likely narrate
- Weighted similarity score (0.0 = different, 1.0 = same)

Tunable: `session.similarity_threshold = 0.7`

## Rate Limiting

Built-in protection against API abuse:

```python
# Default: 10 calls per 60 seconds for GeminiLiveNarrator
# Default: 20 calls per 60 seconds for GeminiLiveSession

# When rate limited, raises:
RuntimeError("Rate limited. Try again in X.Xs")
```

Cache hits don't count against the rate limit.

## What Works

1. **Vertex AI Live API** - Full audio generation working
2. **Smart Narration** - Only speaks when scene changes significantly
3. **Rate Limiting** - Prevents API abuse
4. **Caching** - Repeated scenes return instantly
5. **Sync Wrapper** - Works from both sync and async contexts
6. **Server Audio Playback** - Testing via sounddevice

## Pending / Future

1. **Phone Audio Input** - `answer_question()` is ready, needs phone integration
2. **LLM-based Narration Decision** - `_llm_should_narrate()` placeholder exists
3. **Audio Streaming to Client** - Currently sends full audio after generation
4. **Persistent Session in Server** - server.py still uses per-request mode

## Testing

```bash
# Full service test (working!)
python test_gemini_service.py

# With custom image
python test_gemini_service.py /path/to/image.jpg

# Play generated audio
aplay test_output.wav
```

## Cost Estimate

For `gemini-live-2.5-flash-native-audio` on Vertex AI:

| Usage | Estimated Cost |
|-------|----------------|
| 50 calls (~150s audio) | ~$0.04 - $0.10 |
| 1000 calls | ~$1 - $2 |

Flash models are cheap. Rate limiting is more about preventing spam than cost.

## Resources

- [Gemini Live API Docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api)
- [Vertex AI vs AI Studio](https://ai.google.dev/gemini-api/docs/migrate-to-cloud)
- [google-genai Python SDK](https://github.com/google/generative-ai-python)

## Notes from Development

- The `google.generativeai` package is deprecated → use `google.genai`
- Live API requires Vertex AI, not the basic Generative Language API
- Vertex AI uses OAuth/service accounts, not simple API keys
- Model: `gemini-live-2.5-flash-native-audio` outputs audio only (no text transcript)
- `asyncio.run()` can't be called from running loop → use ThreadPoolExecutor workaround
- Exact scene hashing doesn't work with real camera footage → use semantic similarity
