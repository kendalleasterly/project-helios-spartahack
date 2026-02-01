# Project Helios - Gemini AI Architecture

## Overview

Project Helios uses a **dual-pipeline architecture** with Google's Gemini 2.5 Flash model to provide intelligent, context-aware assistance for blind users. The system is designed to understand surroundings continuously while maintaining conversational context, enabling natural interaction without constant narration.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Dual-Pipeline Design](#dual-pipeline-design)
- [Vision Pipeline](#vision-pipeline)
- [Conversation Pipeline](#conversation-pipeline)
- [Circular Context Architecture](#circular-context-architecture)
- [Model Selection](#model-selection)
- [Data Flow](#data-flow)
- [API Integration](#api-integration)
- [Configuration](#configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     BLIND ASSISTANT SERVICE                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         VISION PIPELINE (Continuous @ 1 FPS)         │  │
│  │  Frame + YOLO → Gemini 2.5 Flash → Spatial Memory   │  │
│  │           ↑                              ↓           │  │
│  │           └──────── Circular Cache ──────┘           │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                │
│                    Spatial Memory Cache                     │
│                    (30-60 seconds)                          │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      CONVERSATION PIPELINE (On-Demand)               │  │
│  │  User Question + Spatial Context → Gemini 2.5 Flash │  │
│  │  → Contextual Answer                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Intelligent Silence**: Only speaks when necessary (safety, major changes, or user questions)
2. **Spatial Memory**: Maintains 30-60 seconds of scene history for contextual awareness
3. **Circular Feedback**: Vision model sees its own recent observations to avoid repetition
4. **Unified Context**: Both pipelines share spatial memory for coherent understanding
5. **Low Latency**: Text responses only (no audio streaming), using device TTS

---

## Dual-Pipeline Design

### Why Two Pipelines?

The system separates **continuous monitoring** from **conversational interaction** to solve the audio chunking problem:

- **Problem**: Camera sends frames @ 1 FPS, but conversations last 5-10+ seconds
- **Solution**: Vision pipeline runs independently, conversation pipeline accesses its history

### Pipeline Comparison

| Feature | Vision Pipeline | Conversation Pipeline |
|---------|----------------|----------------------|
| **Trigger** | Every frame (~1 second) | User asks a question |
| **Input** | Frame + YOLO + Vision History | Question + Spatial Context |
| **Output** | SILENT / SPEAK (safety) | Always responds |
| **Context** | Last 10s of vision | Last 30s of spatial memory |
| **Purpose** | Build spatial memory | Answer questions |

---

## Vision Pipeline

### Purpose
Continuous monitoring at 1 FPS to build spatial memory and provide safety alerts.

### Flow

```
Frame (1 FPS) → YOLO Detection → Vision Model
                                      ↓
                              Check Recent History
                                      ↓
                              DECISION: SPEAK or SILENT?
                                      ↓
                              Store in Spatial Memory
                                      ↓
                              (Optional) Alert User
```

### When Vision Speaks

The vision model only speaks when:
- ✅ **Emergency/Safety**: Vehicle close, obstacle immediate, hazard detected
- ✅ **Major Scene Change**: Entered new room, major layout change, new object type
- ✅ **Navigation Guidance**: Clear path, obstacle ahead, direction change
- ❌ **Scene Unchanged**: Same room, same objects, minor movements

### Circular Feedback Mechanism

The vision model sees its own recent observations to make better decisions:

```python
# Example prompt the vision model receives:

📜 RECENT HISTORY:
[5s ago] Hallway with chairs on both sides
[3s ago] Same hallway, user walking slowly
[1s ago] Same hallway, chair on left

Scene: Hallway with chairs and table
Objects: chair (left, 3 feet), table (center, 5 feet)
👤 User is walking (monitoring mode, no question)
```

**Result**: Vision model says "SILENT: No changes" instead of repeating itself.

### Implementation

```python
# Vision Pipeline (server.py)
response_text = await assistant.process_frame(frame_base64, yolo_results)

if response_text:
    # Vision model decided to speak
    await sio.emit('text_response', {
        'text': response_text,
        'mode': 'vision',
        'emergency': emergency_stop
    }, room=sid)
else:
    # Silent - nothing important
    logger.info("Vision: SILENT")
```

---

## Conversation Pipeline

### Purpose
On-demand question answering with access to spatial memory (last 30-60 seconds).

### Flow

```
User Question → Transcribe Audio → Conversation Model
                                          ↓
                                   Inject Spatial Context
                                   (Objects seen last 30s)
                                          ↓
                                   Generate Answer
                                          ↓
                                   Return Text Response
```

### Spatial Context Injection

When a user asks a question, the conversation model receives:

```python
CURRENT SCENE: Hallway with furniture on both sides

OBJECTS SEEN (last 30 seconds):
  • chair: 2 instance(s)
    - left, 3 feet (25s ago)
    - right, 8 feet (10s ago)
  • table: 1 instance(s)
    - center, 5 feet (20s ago)
  • couch: 1 instance(s)
    - right, 10 feet (5s ago)

TRACKED: 30 observations over 30 seconds

USER QUESTION: "Where can I sit?"
```

**Result**: "I've seen two chairs - one on your left about 3 feet away, and another on your right about 8 feet ahead. There's also a couch on the right at 10 feet."

### Implementation

```python
# Conversation Pipeline (server.py)
if user_question:
    response_text = await assistant.process_user_speech(user_question)

    await sio.emit('text_response', {
        'text': response_text,
        'mode': 'conversation',
        'emergency': emergency_stop
    }, room=sid)
```

---

## Circular Context Architecture

### The Problem: Context Amnesia

Without circular feedback, the vision model would repeat itself:
```
Frame 1: "Hallway with chairs"
Frame 2: "Hallway with chairs"  ❌ REPETITIVE
Frame 3: "Hallway with chairs"  ❌ ANNOYING
```

### The Solution: Self-Awareness

With circular feedback, the vision model remembers what it said:
```
Frame 1: "Hallway with chairs" (SPEAK)
Frame 2: Vision sees: [1s ago] Hallway with chairs
        → "SILENT: No changes"
Frame 3: Vision sees: [2s ago] Hallway, [1s ago] No changes
        → "SILENT: Same scene"
Frame 30: NEW: Kitchen appears!
        → "SPEAK: Entered kitchen!" (detected change)
```

### Technical Implementation

1. **Store Scene Descriptions**: Each frame's analysis is stored with timestamp
2. **Build History Summary**: Recent 5-10 seconds summarized for vision model
3. **Inject into Prompt**: Vision model receives its own past observations
4. **Smart Decisions**: Model compares current scene to history

### Configuration

```python
ContextConfig(
    vision_history_lookback_seconds=10,  # Circular feedback window
    spatial_lookback_seconds=30,         # Conversation context window
    max_scene_history=60                 # Total memory (60 seconds @ 1 FPS)
)
```

---

## Model Selection

### Why Gemini 2.5 Flash?

We use **`gemini-2.5-flash`** for both pipelines:

| Feature | Why It Matters |
|---------|----------------|
| **Multimodal** | Processes images + text in one API call |
| **Streaming** | Returns tokens incrementally (lower perceived latency) |
| **Fast** | "Flash" variant optimized for speed over size |
| **Stable** | Production-ready (not experimental/preview) |
| **Cost-Effective** | Lower cost than Pro variant |
| **1M Context** | Handles long conversation histories |

### Deprecated Models

- ❌ `gemini-2.0-flash-exp` - Retired March 3, 2026
- ❌ `gemini-2.0-flash` - Retired March 3, 2026
- ✅ `gemini-2.5-flash` - Current stable model

### Why NOT Gemini Live API?

We tried Gemini Live API but encountered issues:
- ❌ Responds after every prompt (too chatty)
- ❌ Doesn't silently accumulate context
- ❌ WebSocket audio adds latency to iPhone
- ✅ Standard API gives us full control over when to speak

---

## Data Flow

### End-to-End Flow

```
┌─────────────┐
│   iPhone    │
│   Camera    │
└──────┬──────┘
       │ 1 FPS
       ↓
┌─────────────┐
│   Server    │
│   (YOLO)    │  Detect objects
└──────┬──────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Gemini 2.5 Flash (Vision Pipeline)  │
│  - Receives: Frame + YOLO + History  │
│  - Decision: SPEAK or SILENT?        │
│  - Stores: Scene description         │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────┐
│  Spatial Memory      │  Rolling 60-second buffer
│  (Scene History)     │
└──────┬───────────────┘
       │
       ↓ (when user asks question)
┌───────────────────────────────────────┐
│  Gemini 2.5 Flash (Conversation)      │
│  - Receives: Question + Spatial Cache │
│  - Returns: Contextual answer         │
└───────┬───────────────────────────────┘
        │
        ↓
┌───────────────┐
│  Text → TTS   │  iPhone device TTS
│  → Speaker    │
└───────────────┘
```

### Frame Processing Timeline

```
Time 0s:  Frame → Vision → "Hallway" → Cache
Time 1s:  Frame → Vision → "Same" (SILENT) → Cache
Time 2s:  Frame → Vision → "Same" (SILENT) → Cache
Time 3s:  Frame → Vision → "Same" (SILENT) → Cache
...
Time 15s: Frame → Vision → "Car approaching!" (SPEAK) → Cache → User alerted
...
Time 30s: User asks "Where can I sit?"
          → Conversation gets cache (0s-30s)
          → "I've seen two chairs..."
```

---

## API Integration

### Server Endpoints

#### `video_frame_streaming` (Primary Endpoint)

**Socket.IO Event**: `video_frame_streaming`

**Input**:
```json
{
  "frame": "base64_encoded_jpeg",
  "user_question": "Where can I sit?" | null,
  "debug": false
}
```

**Output** (when there's a response):
```json
{
  "event": "text_response",
  "data": {
    "text": "Chair 3 feet ahead on your left",
    "mode": "vision" | "conversation",
    "emergency": false
  }
}
```

**Modes**:
- **Vision Mode** (`user_question: null`): Continuous monitoring
- **Conversation Mode** (`user_question: "..."`): Question answering

### Client Implementation

#### Vision Pipeline (Continuous)
```javascript
// Send frames at 1 FPS
setInterval(() => {
  socket.emit('video_frame_streaming', {
    frame: captureFrame(),
    user_question: null,  // Vision mode
    debug: false
  });
}, 1000);
```

#### Conversation Pipeline (On-Demand)
```javascript
// When user speaks
const transcription = await transcribeAudio(recordedAudio);
socket.emit('video_frame_streaming', {
  frame: latestFrame,
  user_question: transcription,  // Conversation mode
  debug: false
});
```

#### Response Handling
```javascript
socket.on('text_response', (data) => {
  console.log(`[${data.mode}] ${data.text}`);

  if (data.text) {
    // Send to device TTS
    speakText(data.text, {
      rate: data.emergency ? 1.5 : 1.0  // Faster for emergencies
    });
  }
});
```

---

## Configuration

### ContextConfig Parameters

```python
@dataclass
class ContextConfig:
    # Conversation context window (default: 30s)
    spatial_lookback_seconds: int = 30

    # Vision circular feedback window (default: 10s)
    vision_history_lookback_seconds: int = 10

    # Maximum scene snapshots (default: 60 = 60 seconds @ 1 FPS)
    max_scene_history: int = 60

    # Priority objects for conversation summaries
    priority_objects: List[str] = [
        'chair', 'couch', 'bench', 'sofa',        # Seating
        'stairs', 'door', 'elevator',             # Navigation
        'car', 'truck', 'bicycle', 'motorcycle',  # Vehicles (safety)
        'phone', 'laptop', 'wallet', 'keys',      # Personal items
        'person', 'dog', 'cat'                    # Living beings
    ]

    # Store actual frame images (default: False for memory optimization)
    store_frames: bool = False
```

### Tuning Recommendations

| Use Case | Vision Lookback | Spatial Lookback | Max History |
|----------|----------------|------------------|-------------|
| **Indoor Navigation** | 10s | 30s | 60 |
| **Outdoor Safety** | 5s | 20s | 40 |
| **Object Search** | 10s | 60s | 120 |
| **Memory Constrained** | 5s | 15s | 30 |

---

## System Prompt

The Gemini models receive this system instruction:

```
You are a real-time navigation assistant for a blind person wearing a camera.

You receive:
1. Camera image
2. YOLO object detection data (summary, objects with positions/distances)
3. Recent history (your own past observations from the last 10 seconds)
4. Optional user question

IMPORTANT: Use the recent history to inform your decisions. If you recently
described something and nothing has changed, stay SILENT. If the scene has
changed significantly from your recent observations, SPEAK.

DECISION RULES - When to SPEAK vs stay SILENT:

SPEAK when:
- User asked a question (ALWAYS respond)
- Emergency/safety issue (vehicle close, obstacle immediate, hazard)
- Scene changed significantly (entered new room, major layout change, new object type)
- User requested action ("find my phone", "where can I sit", "read this")
- Important navigation guidance (clear path, obstacle ahead, direction change)

SILENT when:
- Scene nearly identical to what you just described
- Only minor object movements
- Nothing urgent, actionable, or interesting
- Same room, same objects, same layout

OUTPUT FORMAT - CRITICAL:
- Start EVERY response with either "SPEAK: " or "SILENT: " (include the space after colon)
- After the prefix, provide your message
- Keep messages under 20 words unless critical
- Be direct and spatial: "car approaching left", "chair 3 feet ahead"
- Use present tense
```

---

## Performance Characteristics

### Latency Breakdown

| Stage | Typical Time | Notes |
|-------|-------------|-------|
| Frame Capture | ~16ms | iPhone camera |
| Network Upload | ~50-200ms | Depends on connection |
| YOLO Inference | ~100-200ms | GPU (RTX 4070) |
| Gemini API Call | ~500-1500ms | First token latency |
| Text Streaming | ~50ms/token | Subsequent tokens |
| Device TTS | ~100-500ms | iOS native TTS |
| **Total (Vision)** | **~1-2 seconds** | From frame to speech |
| **Total (Conversation)** | **~2-3 seconds** | From question to answer |

### Memory Usage

- **Scene History (60s)**: ~5-10 MB (without stored frames)
- **With Stored Frames**: ~500 MB (not recommended)
- **Gemini Context**: ~50 KB per conversation turn

### Throughput

- **Vision Pipeline**: 1 FPS (1 frame per second)
- **Conversation Pipeline**: On-demand (as fast as user speaks)
- **Concurrent Users**: Tested with 1 user (designed for single-user use)

---

## Debugging

### Spatial Memory Inspection

```python
# Get current spatial memory state
summary = assistant.get_spatial_memory_summary()

print(summary)
# {
#   'total_snapshots': 45,
#   'time_span_seconds': 44.2,
#   'oldest_snapshot_age': 44.2,
#   'has_current_frame': True,
#   'config': {
#     'lookback_seconds': 30,
#     'max_history': 60,
#     'store_frames': False
#   }
# }
```

### Debug Mode

Enable debug frames to see annotated YOLO detections:

```javascript
socket.emit('video_frame_streaming', {
  frame: frameBase64,
  user_question: null,
  debug: true  // Enables debug frame
});

socket.on('debug_frame', (data) => {
  displayImage(data.frame);  // Annotated frame with bounding boxes
  console.log(data.summary);
  console.log(data.object_count);
  console.log(data.mode);  // 'vision' or 'conversation'
});
```

---

## Future Improvements

### Potential Enhancements

1. **Multi-Frame Analysis**: Store frames and analyze motion/trajectories
2. **Voice Activity Detection**: Detect when user is speaking without manual trigger
3. **Context Pruning**: Intelligent removal of redundant spatial data
4. **Semantic Clustering**: Group related objects for better scene understanding
5. **User Preference Learning**: Adapt verbosity based on user feedback
6. **Multi-Language Support**: Support for non-English languages
7. **Offline Mode**: Local model fallback when internet unavailable

### Known Limitations

- **1 FPS Limit**: Fast-moving objects may be missed between frames
- **Single User**: Not designed for multi-user concurrent sessions
- **Internet Dependent**: Requires stable connection to Google Cloud
- **English Only**: System prompt and responses optimized for English
- **Memory Bounded**: 60-second history limit (configurable)

---

## Related Files

- `contextual_gemini_service.py` - Dual-pipeline implementation
- `server.py` - Socket.IO server and YOLO integration
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (GOOGLE_CLOUD_PROJECT, etc.)

---

## License

This architecture is part of Project Helios (SpartaHack 2026).

---

**Last Updated**: January 31, 2026
**Gemini Model**: gemini-2.5-flash (stable)
**Architecture**: Dual-Pipeline with Circular Feedback
