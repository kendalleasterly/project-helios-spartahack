"""
Contextual Gemini Service with Dual-Pipeline Architecture
- Vision Pipeline: Continuous monitoring (1 FPS) with spatial memory
- Conversation Pipeline: On-demand queries with access to vision history
Uses standard Gemini API (generate_content_stream) for decision-making and text streaming.
"""
import base64
import os
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, List, Dict, Any
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, GenerateContentConfig

load_dotenv()

SYSTEM_PROMPT = """You are a real-time navigation assistant for a blind person wearing a camera.

You receive:
1. Camera image
2. YOLO object detection data (summary, objects with positions/distances)
3. Recent history (your own past observations from the last 10 seconds)
4. Optional user question

IMPORTANT: Use the recent history to inform your decisions. If you recently described something and nothing has changed, stay SILENT. If the scene has changed significantly from your recent observations, SPEAK.

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

Examples:
✓ "SPEAK: Car approaching fast on your left, stop now!"
✓ "SPEAK: Empty chair directly ahead, about 5 feet"
✓ "SPEAK: Your phone is on the table to your right"
✓ "SILENT: Same classroom, no significant changes"
✗ "The scene shows..." (WRONG - missing prefix!)

Remember: You must maintain context from previous messages to avoid repeating yourself."""


@dataclass
class StreamedResponse:
    """Result from streaming Gemini call"""
    should_speak: bool
    full_text: str
    latency_ms: float
    total_ms: float


class GeminiContextualNarrator:
    """
    Contextual Gemini narrator using standard API with streaming.
    Maintains conversation history and makes smart decisions about when to speak.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_context_messages: int = 20
    ):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment")

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location
        )
        self.model = model
        self.context_history = []
        self.max_context_messages = max_context_messages

    def _decode_image(self, image_data: str) -> bytes:
        """Decode base64 image, handling data URL prefix."""
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)

    def _build_context_prompt(
        self,
        scene_analysis: dict,
        user_question: Optional[str] = None
    ) -> str:
        """Build text prompt from scene analysis and user question."""
        summary = scene_analysis.get("summary", "Nothing detected")
        emergency = scene_analysis.get("emergency_stop", False)
        objects = scene_analysis.get("objects", [])
        recent_history = scene_analysis.get("recent_history", None)  # NEW: Vision cache

        parts = []

        # Recent history (for vision model context)
        if recent_history:
            parts.append(f"📜 RECENT HISTORY:\n{recent_history}\n")

        # Emergency flag
        if emergency:
            parts.append("🚨 EMERGENCY: Vehicle detected very close!")

        # Scene summary
        parts.append(f"Scene: {summary}")

        # Object details (top 5)
        if objects:
            details = []
            for obj in objects[:5]:
                label = obj.get("label", "object")
                pos = obj.get("position", "unknown")
                dist = obj.get("distance", "unknown")
                details.append(f"{label} ({pos}, {dist})")
            parts.append(f"Objects: {', '.join(details)}")

        # User question
        if user_question:
            parts.append(f"\n👤 USER QUESTION: \"{user_question}\"")
        else:
            parts.append("\n👤 User is walking (monitoring mode, no question)")

        return "\n".join(parts)

    def _add_to_history(self, role: str, content: str):
        """Add message to history and trim if needed."""
        self.context_history.append({"role": role, "content": content})

        # Trim old messages if exceeding limit
        if len(self.context_history) > self.max_context_messages:
            # Keep most recent messages
            self.context_history = self.context_history[-self.max_context_messages:]

    async def process_streaming(
        self,
        scene_analysis: dict,
        image_base64: str,
        user_question: Optional[str] = None
    ) -> AsyncGenerator[tuple[bool, str], None]:
        """
        Process input and stream response tokens.

        Yields:
            (should_speak, text_chunk) tuples
            - First yield contains the decision (parsed from SPEAK:/SILENT: prefix)
            - Subsequent yields contain text chunks (with prefix removed)
        """
        start_time = time.perf_counter()

        # Build prompt
        context_prompt = self._build_context_prompt(scene_analysis, user_question)
        image_bytes = self._decode_image(image_base64)
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Build messages with history
        messages = []

        # Add conversation history
        for msg in self.context_history:
            messages.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

        # Add current input
        messages.append({
            "role": "user",
            "parts": [image_part, {"text": context_prompt}]
        })

        # Call Gemini with streaming
        config = GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=150
        )

        response_stream = self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=messages,
            config=config
        )

        first_chunk_time = None
        should_speak = None
        full_response = ""
        prefix_removed = False

        async for chunk in response_stream:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()

                full_response += chunk.text

                # Parse decision from first chunks
                if should_speak is None:
                    # Check if we have enough to determine SPEAK: or SILENT:
                    if full_response.startswith("SPEAK:"):
                        should_speak = True
                        # Remove prefix and yield clean text
                        clean_text = full_response[6:].lstrip()
                        yield (True, clean_text)
                        prefix_removed = True
                    elif full_response.startswith("SILENT:"):
                        should_speak = False
                        # Remove prefix
                        clean_text = full_response[7:].lstrip()
                        yield (False, clean_text)
                        # Don't stream further if silent (save tokens/latency)
                        break
                    # Keep buffering if we don't have the prefix yet
                    continue
                else:
                    # Already determined decision, stream subsequent chunks
                    yield (should_speak, chunk.text)

        end_time = time.perf_counter()

        # Update history
        self._add_to_history("user", context_prompt)
        self._add_to_history("model", full_response)

    async def process_input(
        self,
        scene_analysis: dict,
        image_base64: str,
        user_question: Optional[str] = None
    ) -> StreamedResponse:
        """
        Non-streaming version that returns complete response.
        Useful for testing or when streaming isn't needed.

        Returns:
            StreamedResponse with decision and full text
        """
        start_time = time.perf_counter()

        should_speak = False
        full_text = ""
        first_chunk_time = None

        async for is_speaking, text_chunk in self.process_streaming(
            scene_analysis, image_base64, user_question
        ):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            should_speak = is_speaking
            full_text += text_chunk

        end_time = time.perf_counter()
        latency = ((first_chunk_time or end_time) - start_time) * 1000
        total = (end_time - start_time) * 1000

        return StreamedResponse(
            should_speak=should_speak,
            full_text=full_text.strip(),
            latency_ms=latency,
            total_ms=total
        )


# ============================================================================
# DUAL-PIPELINE ARCHITECTURE
# ============================================================================

@dataclass
class SceneSnapshot:
    """A moment in time - what the vision system observed"""
    timestamp: float
    yolo_objects: Dict[str, Any]
    scene_description: str
    frame_base64: Optional[str] = None  # Optional: store actual frames


@dataclass
class ContextConfig:
    """Configuration for spatial context management"""
    # How far back to look for spatial context in CONVERSATION queries (seconds)
    spatial_lookback_seconds: int = 30

    # How far back to look for VISION history in circular feedback (seconds)
    # Shorter window for vision processing to keep prompts compact
    vision_history_lookback_seconds: int = 10

    # Maximum number of scene snapshots to keep in memory
    max_scene_history: int = 60  # 60 seconds @ 1 FPS

    # Object types to prioritize in spatial summaries
    priority_objects: List[str] = field(default_factory=lambda: [
        'chair', 'couch', 'bench', 'sofa',  # Seating
        'stairs', 'door', 'elevator',  # Navigation
        'car', 'truck', 'bicycle', 'motorcycle',  # Vehicles (safety)
        'phone', 'laptop', 'wallet', 'keys',  # Personal items
        'person', 'dog', 'cat'  # Living beings
    ])

    # Whether to store actual frame images (memory intensive)
    store_frames: bool = False

    # Maximum scene summaries to include in context
    max_scene_summaries: int = 10


class BlindAssistantService:
    """
    Orchestrates dual-pipeline architecture for blind assistance:

    1. Vision Pipeline: Continuous monitoring (1 FPS) building spatial memory
    2. Conversation Pipeline: On-demand queries with vision history access

    Vision context flows one-way into conversations, enabling questions like
    "Where can I sit?" to reference objects seen in the past 30-60 seconds.
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize the dual-pipeline assistant.

        Args:
            config: Configuration for context management (uses defaults if None)
        """
        self.config = config or ContextConfig()

        # Vision monitoring narrator - maintains scene understanding
        self.vision_narrator = GeminiContextualNarrator(
            model="gemini-2.5-flash",
            max_context_messages=20  # Internal vision model context
        )

        # Conversation narrator - handles user queries
        self.conversation_narrator = GeminiContextualNarrator(
            model="gemini-2.5-flash",
            max_context_messages=10  # Recent conversation history
        )

        # Spatial memory: rolling buffer of scene snapshots
        self.scene_history: deque[SceneSnapshot] = deque(
            maxlen=self.config.max_scene_history
        )

        # Cache latest frame for conversation queries
        self.latest_frame: Optional[str] = None
        self.latest_yolo: Optional[Dict[str, Any]] = None

    async def process_frame(
        self,
        frame_base64: str,
        yolo_objects: Dict[str, Any]
    ) -> Optional[str]:
        """
        Vision Pipeline: Process incoming frame (called every ~1 second).

        Builds spatial memory and occasionally speaks for safety/major changes.

        CIRCULAR ARCHITECTURE: Vision cache feeds back into vision processing,
        allowing the model to see its recent history and make better decisions.

        Args:
            frame_base64: Base64-encoded frame image
            yolo_objects: YOLO detection results with summary, objects, emergency flag

        Returns:
            Text to speak if vision model decides to (None if SILENT)
        """
        # Update cache
        self.latest_frame = frame_base64
        self.latest_yolo = yolo_objects

        # Check for immediate danger (local rules can override for instant response)
        immediate_danger = self._check_immediate_danger(yolo_objects)
        if immediate_danger:
            # Store in history but return immediately
            snapshot = SceneSnapshot(
                timestamp=time.time(),
                yolo_objects=yolo_objects,
                scene_description=immediate_danger,
                frame_base64=frame_base64 if self.config.store_frames else None
            )
            self.scene_history.append(snapshot)
            return immediate_danger

        # CIRCULAR: Build vision history summary for vision model
        vision_history = self._build_vision_history_context()

        # Add vision history to scene analysis (circular feedback)
        scene_analysis_with_history = {
            **yolo_objects,  # Include all YOLO data
            "recent_history": vision_history  # Add vision cache
        }

        # Vision model analyzes scene WITH its own recent history
        response = await self.vision_narrator.process_input(
            scene_analysis=scene_analysis_with_history,
            image_base64=frame_base64,
            user_question=None
        )

        # Store snapshot in spatial memory
        snapshot = SceneSnapshot(
            timestamp=time.time(),
            yolo_objects=yolo_objects,
            scene_description=response.full_text,
            frame_base64=frame_base64 if self.config.store_frames else None
        )
        self.scene_history.append(snapshot)

        # Return text if vision model wants to speak
        if response.should_speak:
            return response.full_text

        return None

    async def process_user_speech(
        self,
        user_question: str
    ) -> str:
        """
        Conversation Pipeline: Handle user question (called on-demand).

        Injects spatial context from vision history before querying conversation model.

        Args:
            user_question: Transcribed user speech/question

        Returns:
            Text response to speak to user
        """
        if not self.latest_frame:
            return "I haven't seen anything yet. Please wait a moment."

        # Build enriched spatial context from vision history
        spatial_context = self._build_spatial_context()

        # Create scene analysis with spatial context
        scene_analysis = {
            "summary": spatial_context,
            "objects": self.latest_yolo.get("objects", []) if self.latest_yolo else [],
            "emergency_stop": False  # User questions aren't emergencies
        }

        # Conversation model gets vision context + question
        response = await self.conversation_narrator.process_input(
            scene_analysis=scene_analysis,
            image_base64=self.latest_frame,
            user_question=user_question
        )

        return response.full_text

    def _build_vision_history_context(self) -> str:
        """
        Build compact vision history for CIRCULAR feedback to vision model.

        This allows the vision model to see its own recent observations when
        processing new frames, enabling it to:
        - Detect scene changes ("was hallway, now kitchen")
        - Avoid repetition ("already mentioned chair")
        - Track trends ("vehicle getting closer")

        Returns:
            Compact summary of recent vision observations
        """
        if not self.scene_history:
            return "First observation."

        # Get recent scenes (shorter window than conversation context)
        cutoff_time = time.time() - self.config.vision_history_lookback_seconds
        recent_scenes = [
            scene for scene in self.scene_history
            if scene.timestamp >= cutoff_time
        ]

        if not recent_scenes:
            recent_scenes = [self.scene_history[-1]]

        # If only one scene, this is likely the first few frames
        if len(recent_scenes) == 1:
            return f"Previous: {recent_scenes[0].scene_description}"

        # Build compact history (last 3-5 observations with timing)
        history_items = []
        for scene in recent_scenes[-5:]:  # Last 5 scenes max
            seconds_ago = int(time.time() - scene.timestamp)
            if seconds_ago == 0:
                time_str = "just now"
            elif seconds_ago == 1:
                time_str = "1s ago"
            else:
                time_str = f"{seconds_ago}s ago"

            # Extract just the description (remove SPEAK:/SILENT: prefix if present)
            desc = scene.scene_description
            if desc.startswith("SPEAK: "):
                desc = desc[7:].strip()
            elif desc.startswith("SILENT: "):
                desc = desc[8:].strip()

            history_items.append(f"[{time_str}] {desc}")

        return "\n".join(history_items)

    def _check_immediate_danger(self, yolo_objects: Dict[str, Any]) -> Optional[str]:
        """
        Check for immediate safety hazards using simple rules.

        This provides instant alerts (<100ms) for critical situations before
        the vision model even processes the frame.

        Args:
            yolo_objects: YOLO detection results

        Returns:
            Alert message if danger detected, None otherwise
        """
        if yolo_objects.get("emergency_stop"):
            return "STOP! Vehicle approaching!"

        # Check for very close vehicles
        objects = yolo_objects.get("objects", [])
        for obj in objects:
            label = obj.get("label", "")
            distance = obj.get("distance", "")

            # Vehicle very close
            if label in ["car", "truck", "bus", "motorcycle"] and "close" in distance.lower():
                position = obj.get("position", "nearby")
                return f"STOP! {label.title()} {position}!"

            # Stairs very close
            if label == "stairs" and any(term in distance.lower() for term in ["close", "1 foot", "2 foot"]):
                position = obj.get("position", "ahead")
                return f"Careful! Stairs {position}!"

        return None

    def _build_spatial_context(self) -> str:
        """
        Build spatial context summary from recent vision history.

        This is the key function that enables conversation queries like
        "Where can I sit?" to reference objects seen over the past 30+ seconds.

        Returns:
            Formatted spatial context string for conversation model
        """
        if not self.scene_history:
            return "No spatial data available yet."

        # Get recent scenes within lookback window
        cutoff_time = time.time() - self.config.spatial_lookback_seconds
        recent_scenes = [
            scene for scene in self.scene_history
            if scene.timestamp >= cutoff_time
        ]

        if not recent_scenes:
            # Lookback too short, use at least the latest scene
            recent_scenes = [self.scene_history[-1]]

        # Build context parts
        context_parts = []

        # 1. Current scene (most recent)
        current_scene = recent_scenes[-1]
        context_parts.append(f"CURRENT SCENE: {current_scene.scene_description}")

        # 2. Object inventory from recent history
        objects_seen = self._collect_objects_from_scenes(recent_scenes)
        if objects_seen:
            context_parts.append("\nOBJECTS SEEN (last 30 seconds):")
            for label, instances in objects_seen.items():
                count = len(instances)
                context_parts.append(f"  • {label}: {count} instance(s)")

                # Show details for priority objects
                if label in self.config.priority_objects:
                    for inst in instances[:3]:  # Max 3 per type
                        age = inst['seconds_ago']
                        age_str = f"{age}s ago" if age > 0 else "now"
                        context_parts.append(
                            f"    - {inst['position']}, {inst['distance']} ({age_str})"
                        )

        # 3. Scene evolution summary
        if len(recent_scenes) > 5:
            context_parts.append(
                f"\nTRACKED: {len(recent_scenes)} observations over {self.config.spatial_lookback_seconds} seconds"
            )

        return "\n".join(context_parts)

    def _collect_objects_from_scenes(
        self,
        scenes: List[SceneSnapshot]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect and organize all objects seen across multiple scenes.

        Args:
            scenes: List of scene snapshots to analyze

        Returns:
            Dictionary mapping object labels to lists of instances with details
        """
        objects_seen: Dict[str, List[Dict[str, Any]]] = {}
        current_time = time.time()

        for scene in scenes:
            scene_objects = scene.yolo_objects.get('objects', [])
            for obj in scene_objects:
                label = obj.get('label', 'unknown')

                # Initialize list for this object type
                if label not in objects_seen:
                    objects_seen[label] = []

                # Add instance with timing info
                objects_seen[label].append({
                    'position': obj.get('position', 'unknown'),
                    'distance': obj.get('distance', 'unknown'),
                    'seconds_ago': int(current_time - scene.timestamp)
                })

        # Sort by priority (priority objects first)
        sorted_objects = {}

        # Priority objects first
        for label in self.config.priority_objects:
            if label in objects_seen:
                sorted_objects[label] = objects_seen[label]

        # Then everything else
        for label in objects_seen:
            if label not in sorted_objects:
                sorted_objects[label] = objects_seen[label]

        return sorted_objects

    def get_spatial_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current spatial memory state.

        Useful for debugging and monitoring.

        Returns:
            Dictionary with spatial memory statistics
        """
        return {
            "total_snapshots": len(self.scene_history),
            "time_span_seconds": (
                self.scene_history[-1].timestamp - self.scene_history[0].timestamp
                if len(self.scene_history) > 1 else 0
            ),
            "oldest_snapshot_age": (
                time.time() - self.scene_history[0].timestamp
                if self.scene_history else 0
            ),
            "has_current_frame": self.latest_frame is not None,
            "config": {
                "lookback_seconds": self.config.spatial_lookback_seconds,
                "max_history": self.config.max_scene_history,
                "store_frames": self.config.store_frames
            }
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
Example usage of the dual-pipeline architecture:

```python
import asyncio
from contextual_gemini_service import BlindAssistantService, ContextConfig

async def main():
    # Initialize service with custom config (optional)
    config = ContextConfig(
        spatial_lookback_seconds=30,
        max_scene_history=60,
        store_frames=False  # Set True if you need multi-frame analysis
    )
    assistant = BlindAssistantService(config)

    # ========================================
    # VISION PIPELINE (runs continuously @ 1 FPS)
    # ========================================
    async def vision_loop():
        while True:
            # Get frame from phone camera
            frame = await get_camera_frame()  # Your implementation

            # Get YOLO detections
            yolo_results = await run_yolo(frame)  # Your implementation
            # Expected format: {
            #     "summary": "Hallway with furniture",
            #     "objects": [
            #         {"label": "chair", "position": "left", "distance": "3 feet"},
            #         {"label": "table", "position": "center", "distance": "5 feet"}
            #     ],
            #     "emergency_stop": False
            # }

            # Process frame (builds spatial memory)
            response = await assistant.process_frame(frame, yolo_results)

            # If vision model wants to speak (safety/major changes)
            if response:
                await speak_to_user(response)  # Your TTS implementation

            await asyncio.sleep(1.0)  # 1 FPS

    # ========================================
    # CONVERSATION PIPELINE (on-demand)
    # ========================================
    async def conversation_loop():
        while True:
            # Wait for user to speak (VAD detects speech)
            audio = await wait_for_user_speech()  # Your implementation

            # Transcribe audio
            transcription = await transcribe_audio(audio)  # Your implementation

            # Process question (gets vision history context)
            response = await assistant.process_user_speech(transcription)

            # Speak response
            await speak_to_user(response)  # Your TTS implementation

    # Run both pipelines concurrently
    await asyncio.gather(
        vision_loop(),
        conversation_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Key points:
- Vision pipeline runs continuously (1 FPS), building spatial memory
- Conversation pipeline runs on-demand when user speaks
- Vision context automatically flows into conversations
- User can ask about things seen in the past 30-60 seconds
- Both use standard Gemini API (no websockets, text responses only)
"""
