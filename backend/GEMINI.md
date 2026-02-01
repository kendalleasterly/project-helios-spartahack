# Project Helios - Gemini AI Architecture v2

## Overview

Project Helios uses a **heuristic-driven dual-pipeline architecture** with Google's Gemini Flash model to provide intelligent, context-aware navigation assistance for blind users.

**Key Change from v1**: Instead of relying on Gemini to decide SPEAK/SILENT (unreliable), we use **YOLO-based heuristics** to decide *when* to call Gemini, and Gemini focuses purely on *what* to say.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Decision Flow: When to Speak](#decision-flow-when-to-speak)
- [Vision Pipeline](#vision-pipeline)
- [Conversation Pipeline](#conversation-pipeline)
- [Heuristics Engine](#heuristics-engine)
- [Helios Personality](#helios-personality)
- [System Prompts](#system-prompts)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Implementation Guide](#implementation-guide)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT HELIOS v2                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    FRAME INPUT (1 FPS)                      │ │
│  │                  Camera → YOLO Detection                    │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              HEURISTICS ENGINE (Fast, Local)                │ │
│  │                                                             │ │
│  │   Input: YOLO detections, distances, positions              │ │
│  │   Output: should_call_gemini (bool), urgency_level          │ │
│  │                                                             │ │
│  │   Rules:                                                    │ │
│  │   • emergency_stop = true → ALWAYS call (URGENT)            │ │
│  │   • distance = "immediate" → ALWAYS call (ALERT)            │ │
│  │   • distance = "close" + center → call (GUIDANCE)           │ │
│  │   • user_question present → ALWAYS call (CONVERSATION)      │ │
│  │   • only "far" objects, no changes → SKIP Gemini            │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              ▼                             ▼                    │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │   SKIP (Silent)     │      │     CALL GEMINI             │  │
│  │                     │      │                             │  │
│  │  • No API call      │      │  Vision Mode:               │  │
│  │  • Update history   │      │  → Proactive navigation     │  │
│  │  • Save bandwidth   │      │  → Actionable guidance      │  │
│  │                     │      │                             │  │
│  │                     │      │  Conversation Mode:         │  │
│  │                     │      │  → Answer user question     │  │
│  │                     │      │  → Always responds          │  │
│  └─────────────────────┘      └──────────────┬──────────────┘  │
│                                              │                  │
│                                              ▼                  │
│                               ┌─────────────────────────────┐  │
│                               │     TTS → User Hears        │  │
│                               └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Heuristics First**: YOLO data decides if we call Gemini (fast, reliable, local)
2. **Proactive Guidance**: Speak up about obstacles without being asked
3. **Actionable Output**: Tell users what to DO, not just what exists
4. **Personality**: Helios is a calm, confident friend - not a robot
5. **Always Answer Questions**: Conversation mode bypasses heuristics

---

## Decision Flow: When to Speak

### The Problem with v1

In v1, we asked Gemini to prefix responses with `SPEAK:` or `SILENT:`. This failed because:
- LLMs are inconsistent with formatting
- Chunked streaming made prefix parsing unreliable
- Model defaulted to SILENT too often
- Added latency waiting for decision

### The v2 Solution: Heuristics

**Decide BEFORE calling Gemini, using YOLO data:**

```
┌─────────────────────────────────────────────────────────────┐
│                    SHOULD WE CALL GEMINI?                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. User asked a question?                                  │
│     YES → Call Gemini (Conversation Mode)                   │
│                                                             │
│  2. Emergency detected (emergency_stop = true)?             │
│     YES → Call Gemini IMMEDIATELY (Urgent)                  │
│                                                             │
│  3. Any object at "immediate" distance (<3 feet)?           │
│     YES → Call Gemini (Alert)                               │
│                                                             │
│  4. Object at "close" distance + in center of frame?        │
│     YES → Call Gemini (Guidance)                            │
│                                                             │
│  5. New important object not seen in last 10 seconds?       │
│     YES → Call Gemini (Info)                                │
│                                                             │
│  6. Path is clear, only far objects, scene unchanged?       │
│     NO  → Skip Gemini, stay silent                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Urgency Levels

| Level | Trigger | Gemini Behavior | Example |
|-------|---------|-----------------|---------|
| **URGENT** | `emergency_stop` or vehicle immediate | Ultra-short, interrupt | "Stop! Car left!" |
| **ALERT** | Any object at immediate distance | Short warning | "Heads up, wall ahead." |
| **GUIDANCE** | Close object in path | Navigation help | "Chair left, keep right." |
| **INFO** | New interesting object | Brief mention | "Door on your right." |
| **CONVERSATION** | User asked question | Full answer | "Yeah, there's a chair about 6 feet ahead." |

---

## Vision Pipeline

### Purpose
Proactive navigation assistance. Speaks when there's something the user needs to know.

### When Vision Speaks (Heuristic Rules)

```python
def should_speak(scene_analysis: dict, recent_history: list) -> tuple[bool, str]:
    """
    Determine if we should call Gemini based on YOLO data.
    Returns (should_call, urgency_level)
    """
    objects = scene_analysis.get("objects", [])
    emergency = scene_analysis.get("emergency_stop", False)

    # Rule 1: Emergency - ALWAYS speak
    if emergency:
        return (True, "URGENT")

    # Rule 2: Immediate distance - ALWAYS speak
    for obj in objects:
        if obj.get("distance") == "immediate":
            return (True, "ALERT")

    # Rule 3: Close + in walking path (center)
    for obj in objects:
        if obj.get("distance") == "close":
            pos = obj.get("position", "")
            if "center" in pos:
                return (True, "GUIDANCE")

    # Rule 4: New important object (not seen recently)
    important_labels = {"door", "stairs", "elevator", "person", "car", "chair"}
    current_labels = {obj.get("label") for obj in objects}
    recent_labels = get_recent_labels(recent_history, seconds=10)

    new_important = current_labels & important_labels - recent_labels
    if new_important:
        return (True, "INFO")

    # Rule 5: Path is clear, nothing new
    return (False, "SILENT")
```

### What Vision Says

When heuristics trigger a Gemini call, the prompt instructs Helios to give **actionable guidance**:

| Situation | Bad (Descriptive) | Good (Actionable) |
|-----------|-------------------|-------------------|
| Obstacle ahead | "There is a chair in front of you" | "Chair ahead, veer right" |
| Person nearby | "A person is detected" | "Someone on your left, passing by" |
| Clear path | "The hallway is empty" | "You're good, clear ahead" |
| Door found | "I see a door" | "Door 10 feet ahead, straight shot" |

---

## Conversation Pipeline

### Purpose
Answer user questions. **Always responds** - bypasses heuristics.

### Flow

```
User speaks → Transcription → ALWAYS call Gemini → Answer → TTS
```

### Key Rules

1. **Never silent**: If user asked a question, Helios answers
2. **Use spatial context**: Access last 30 seconds of vision history
3. **Be conversational**: Natural, helpful, not robotic
4. **Be honest**: "I can't quite tell" is better than guessing

### Example Interactions

```
User: "What's in front of me?"
Helios: "Looks like a hallway. Couple chairs along the right wall, door at the far end."

User: "Where can I sit?"
Helios: "There's a chair about 6 feet ahead, slightly left. Want me to guide you there?"

User: "Is anyone here?"
Helios: "Just one person, off to your right. Looks like they're walking away."

User: "What does this sign say?"
Helios: "Hmm, I can see there's text but it's too small to read clearly from here."
```

---

## Heuristics Engine

### Implementation

```python
# heuristics.py

from dataclasses import dataclass
from typing import Optional, List, Set
from enum import Enum

class UrgencyLevel(Enum):
    SILENT = "silent"
    INFO = "info"
    GUIDANCE = "guidance"
    ALERT = "alert"
    URGENT = "urgent"

@dataclass
class SpeakDecision:
    should_speak: bool
    urgency: UrgencyLevel
    reason: str

# Objects that warrant speaking about
IMPORTANT_OBJECTS = {
    # Navigation
    "door", "stairs", "elevator", "escalator",
    # Seating
    "chair", "couch", "bench", "sofa",
    # Safety
    "car", "truck", "bus", "motorcycle", "bicycle",
    # People
    "person",
    # Hazards
    "dog", "cat",
}

# Objects in path require more attention
PATH_POSITIONS = {"center", "mid-center", "bottom-center"}

def evaluate_scene(
    scene_analysis: dict,
    recent_objects: Set[str],
    last_spoke_seconds_ago: float
) -> SpeakDecision:
    """
    Evaluate YOLO scene data and decide if Gemini should be called.

    Args:
        scene_analysis: YOLO detection results
        recent_objects: Object labels seen in last N seconds
        last_spoke_seconds_ago: Time since last speech output

    Returns:
        SpeakDecision with should_speak, urgency, and reason
    """
    objects = scene_analysis.get("objects", [])
    emergency = scene_analysis.get("emergency_stop", False)

    # URGENT: Emergency flag
    if emergency:
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.URGENT,
            reason="Emergency detected"
        )

    # ALERT: Immediate distance (< 3 feet)
    immediate_objects = [o for o in objects if o.get("distance") == "immediate"]
    if immediate_objects:
        labels = [o.get("label") for o in immediate_objects]
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.ALERT,
            reason=f"Immediate: {', '.join(labels)}"
        )

    # GUIDANCE: Close objects in walking path
    close_in_path = [
        o for o in objects
        if o.get("distance") == "close"
        and any(p in o.get("position", "") for p in PATH_POSITIONS)
    ]
    if close_in_path:
        labels = [o.get("label") for o in close_in_path]
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.GUIDANCE,
            reason=f"In path: {', '.join(labels)}"
        )

    # INFO: New important objects (debounced)
    if last_spoke_seconds_ago > 3.0:  # Don't spam
        current_labels = {o.get("label") for o in objects}
        new_important = (current_labels & IMPORTANT_OBJECTS) - recent_objects

        if new_important:
            return SpeakDecision(
                should_speak=True,
                urgency=UrgencyLevel.INFO,
                reason=f"New: {', '.join(new_important)}"
            )

    # SILENT: Nothing noteworthy
    return SpeakDecision(
        should_speak=False,
        urgency=UrgencyLevel.SILENT,
        reason="Clear path, no changes"
    )
```

### Integration with Server

```python
# In server.py frame handler

from heuristics import evaluate_scene, UrgencyLevel

async def handle_frame(frame_data, user_question):
    scene_analysis = run_yolo(frame_data)

    # Conversation mode: always call Gemini
    if user_question:
        response = await gemini.conversation(scene_analysis, user_question)
        return response

    # Vision mode: use heuristics
    decision = evaluate_scene(
        scene_analysis,
        recent_objects=get_recent_objects(),
        last_spoke_seconds_ago=get_time_since_last_speech()
    )

    if not decision.should_speak:
        # Silent - just update history, no API call
        update_scene_history(scene_analysis)
        return None

    # Call Gemini with urgency context
    response = await gemini.vision(
        scene_analysis,
        urgency=decision.urgency
    )

    return response
```

---

## Helios Personality

Helios is a **calm, confident friend** who has your back - not a robotic assistant.

### Voice Characteristics

| Trait | Description | Example |
|-------|-------------|---------|
| **Warm** | Friendly but not patronizing | "You're good, keep going" |
| **Direct** | Gets to the point | "Chair left, go right" |
| **Calm** | Steady even in tense moments | "Heads up, just wait a sec" |
| **Confident** | Knows what they're doing | "Door's straight ahead" |
| **Honest** | Admits uncertainty | "Hard to tell from here" |

### Speech Patterns

**Urgent situations** - Sharp but calm:
- "Whoa, hold up. Stairs."
- "Stop. Car coming left."
- "Wait - door opening."

**Routine guidance** - Casual and brief:
- "You're good, clear ahead."
- "Chair on your right."
- "Someone passing by."

**Answering questions** - Conversational:
- "Yeah, there's a door about 10 feet ahead."
- "Hmm, I can see a sign but can't quite read it."
- "Just one person, off to your left."

**Positive feedback**:
- "Perfect, you've got it."
- "Nice, wide open."
- "Yep, that's the one."

---

## System Prompts

### Vision System Prompt

```python
VISION_SYSTEM_PROMPT = """You are Helios, a sharp-eyed guide for your blind companion. Think of yourself as their trusted spotter - calm, confident, and always looking out for them.

Your personality:
- Warm but not patronizing
- Direct but not robotic
- Confident but not bossy
- Celebrate small wins ("Nice, clear path ahead")
- Stay calm in tense moments ("Heads up, just wait a sec")

You receive:
1. Camera image
2. YOLO detections with positions and distances
3. Urgency level: URGENT, ALERT, GUIDANCE, or INFO
4. Recent observations (last 10 seconds)

HOW TO SPEAK:

For URGENT (emergency):
- Ultra short: "Stop!" "Car left!" "Stairs!"
- 1-4 words max

For ALERT (immediate distance):
- Short warning: "Heads up, wall ahead." "Someone right in front."
- Under 8 words

For GUIDANCE (close, in path):
- Navigation help: "Chair left, keep right." "Table ahead, go around."
- Under 10 words

For INFO (new objects):
- Brief mention: "Door on your right." "Stairs coming up."
- Under 8 words

STYLE RULES:

Give instructions, not descriptions:
- ❌ "There is a chair on your left at 4 feet"
- ✅ "Chair left, you're good."

- ❌ "I detect a person ahead"
- ✅ "Someone ahead, stepping aside."

Be their eyes, not a computer:
- ❌ "Obstacle detected at immediate proximity"
- ✅ "Whoa, hold up. Wall."

When path is clear:
- "You're good, keep going."
- "Clear ahead."
- "Nice, wide open."

OUTPUT: Just speak naturally. No prefixes needed."""
```

### Conversation System Prompt

```python
CONVERSATION_SYSTEM_PROMPT = """You are Helios, a friendly vision assistant for your blind companion. They asked you a question - answer like a helpful friend would.

Your vibe:
- Helpful and direct, not clinical
- Give useful info, not lectures
- Honest if you can't see something clearly
- Keep it conversational

You receive:
1. Camera image
2. YOLO detections with positions/distances
3. Spatial context from recent observations (last 30 seconds)
4. The user's question

HOW TO ANSWER:

Be natural:
- ❌ "I have detected a brown chair positioned at approximately 5 feet."
- ✅ "Yeah, there's a brown chair about 5 feet ahead, a bit to your left."

- ❌ "Affirmative, there is a door in the current field of view."
- ✅ "Yep, door's straight ahead."

- ❌ "I am unable to determine the answer."
- ✅ "Hmm, hard to tell from here."

For location questions:
- Give distance and direction
- Add a quick tip if helpful ("handle's on the right")

For "what's around" questions:
- Hit the highlights, don't list everything
- Focus on what's useful

When unsure:
- Be honest: "I think so, but hard to tell..."
- Never make stuff up

Keep it short, keep it real. You're helping a friend, not writing a report."""
```

---

## Data Flow

### Complete Flow Diagram

```
┌──────────────┐
│   iPhone     │
│   Camera     │
└──────┬───────┘
       │ 1 FPS
       ▼
┌──────────────┐
│   Server     │
│   (YOLO)     │──────────────────────────────────────┐
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌──────────────────────────────────────────────┐     │
│            HEURISTICS ENGINE                  │     │
│                                              │     │
│  if user_question:                           │     │
│      → Conversation Pipeline (always)        │     │
│                                              │     │
│  elif emergency or immediate or close+path:  │     │
│      → Vision Pipeline (with urgency)        │     │
│                                              │     │
│  else:                                       │     │
│      → Silent (no Gemini call)               │     │
└──────┬─────────────────┬─────────────────────┘     │
       │                 │                           │
  [SILENT]          [SPEAK]                          │
       │                 │                           │
       ▼                 ▼                           │
┌────────────┐  ┌─────────────────────────────┐     │
│ Update     │  │      Gemini Flash           │     │
│ History    │  │                             │     │
│ Only       │  │  Vision: Proactive help     │     │
└────────────┘  │  Conversation: Answer Q     │     │
                │                             │     │
                │  + Scene context ◄──────────┼─────┘
                │  + Urgency level            │
                │  + Recent history           │
                └──────────────┬──────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  TTS Output │
                        └─────────────┘
```

### Frame Processing Examples

**Example 1: Clear Hallway (Silent)**
```
Frame → YOLO: [chair (far, left), door (far, center)]
     → Heuristics: No immediate/close objects, seen before
     → Decision: SILENT
     → Result: No Gemini call, update history only
```

**Example 2: Obstacle Ahead (Speak)**
```
Frame → YOLO: [chair (close, center)]
     → Heuristics: Close + in path
     → Decision: SPEAK (GUIDANCE)
     → Gemini: "Chair ahead, go left to pass."
     → TTS → User hears guidance
```

**Example 3: Emergency (Urgent)**
```
Frame → YOLO: [car (immediate, center)], emergency_stop=true
     → Heuristics: Emergency!
     → Decision: SPEAK (URGENT)
     → Gemini: "Stop! Car!"
     → TTS (fast) → User stops
```

**Example 4: User Question (Always Answer)**
```
Frame + Question: "Where's the door?"
     → Heuristics: User question present
     → Decision: SPEAK (CONVERSATION)
     → Gemini: "Door's straight ahead, about 15 feet. Handle will be on your right."
     → TTS → User hears answer
```

---

## Configuration

### Heuristics Tuning

```python
@dataclass
class HeuristicsConfig:
    # Debounce: minimum seconds between INFO-level speech
    info_debounce_seconds: float = 3.0

    # Objects that warrant proactive speaking
    important_objects: set = {
        "door", "stairs", "elevator", "chair", "person", "car"
    }

    # Positions considered "in walking path"
    path_positions: set = {"center", "mid-center", "bottom-center"}

    # Distance thresholds (if using numeric distances)
    immediate_threshold_feet: float = 3.0
    close_threshold_feet: float = 8.0
```

### Gemini Configuration

```python
@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"

    # Higher for vision (faster, less thinking)
    vision_max_tokens: int = 100
    vision_temperature: float = 0.3

    # Higher for conversation (more thoughtful)
    conversation_max_tokens: int = 200
    conversation_temperature: float = 0.7

    # Important: Account for thinking tokens
    # Gemini 2.5+ uses ~200 thinking tokens
    # Set max_tokens = thinking + response
```

---

## Implementation Guide

### Step 1: Implement Heuristics

Create `heuristics.py` with the `evaluate_scene()` function. This runs locally, no API call.

### Step 2: Modify Frame Handler

```python
# Before (v1): Always called Gemini, parsed SPEAK/SILENT
response = await gemini.process(frame, question)
if response.should_speak:
    send_to_tts(response.text)

# After (v2): Heuristics decide, Gemini just speaks
if question:
    # Conversation: always respond
    response = await gemini.conversation(frame, question)
    send_to_tts(response)
elif should_speak_heuristic(scene):
    # Vision: heuristics triggered
    response = await gemini.vision(frame, urgency)
    send_to_tts(response)
else:
    # Silent: no API call
    update_history(scene)
```

### Step 3: Update Prompts

Remove all `SPEAK:/SILENT:` instructions. Gemini now just outputs natural speech.

### Step 4: Test Scenarios

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty hallway | Silent (no Gemini call) |
| Chair 2 feet ahead | Alert: "Heads up, chair ahead" |
| User asks "what's around?" | Full conversational answer |
| Car approaching fast | Urgent: "Stop! Car left!" |
| Same scene for 10 seconds | Silent after first mention |

---

## Migration from v1

### What's Removed
- `SPEAK:/SILENT:` prefix parsing
- Gemini-based speak/silent decisions
- Complex streaming prefix detection

### What's Added
- `heuristics.py` module
- `UrgencyLevel` enum
- Pre-call decision logic
- Urgency-aware prompts

### What's Changed
- System prompts (simpler, no prefix)
- Frame handler flow
- Personality injection (Helios)

---

## Related Files

- `heuristics.py` - Speaking decision engine (NEW)
- `contextual_gemini_service.py` - Gemini API wrapper
- `server.py` - Socket.IO server, YOLO integration
- `prompts.py` - System prompts (optional refactor)

---

**Last Updated**: February 1, 2026
**Architecture Version**: v2 (Heuristics-Driven)
**Gemini Model**: gemini-2.5-flash / gemini-3-flash
**Key Change**: YOLO heuristics decide when to speak, Gemini decides what to say
