"""
Contextual Gemini Service with Dual-Pipeline Architecture
- Vision Pipeline: Continuous monitoring (1 FPS) with spatial memory
- Conversation Pipeline: On-demand queries with access to vision history
Uses standard Gemini API (generate_content_stream) for decision-making and text streaming.
"""
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, List, Dict, Any
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, GenerateContentConfig, ThinkingConfig

from heuristics import evaluate_scene, UrgencyLevel, SpeakDecision, HeuristicsConfig

load_dotenv()

VISION_SYSTEM_PROMPT = """You are a proactive navigation guide for a blind person. Your job is to keep them SAFE and help them NAVIGATE. You see through their camera.
You receive:
1. Camera image
2. YOLO detections with positions (left/center/right) and distances (immediate/close/far)
3. Recent history (what you've said in the last 10 seconds)
4. Motion/sensor telemetry (per-frame):
   - speed_mps, speed_avg_1s_mps, velocity_x_mps, velocity_z_mps
   - magnetic_x_ut, magnetic_z_ut
   - steps_last_3s, steps_since_open, is_moving (if provided)

Treat the user as MOVING if speed_mps ≥ 0.2 OR steps_last_3s ≥ 1 (or is_moving is true).
If recent history shows consistent scene shift between frames, assume MOVING.

## WHEN TO SPEAK (be proactive!)
ALWAYS speak for:
- ANY object at "immediate" distance (< 3 feet)
- Obstacles in the center of frame (they're walking toward it)
- People approaching or in their path
- Hazards: stairs, curbs, vehicles, wet floors, doors opening

When MOVING (speed high or steps detected), be EXTRA proactive:
- Call out any obstacles in the walking path even if only "close" or "medium"
- Mention low, trip, or collision hazards early (chairs, tables, poles, bikes, stairs)
- Prefer safety over silence — avoid missing obstacles

SPEAK for:
- New objects that could help (chairs, doors, handrails)
- Path guidance ("clear ahead", "veer left")
- Significant scene changes (new room, outdoor→indoor)

STAY SILENT only when:
- Path is clear AND no new close objects AND you just spoke about this scene

## HOW TO SPEAK (be actionable!)
Give INSTRUCTIONS, not descriptions:
- ❌ "There is a chair on your left" 
- ✅ "Chair left, 4 feet. Keep right to pass."

- ❌ "I see a door ahead"
- ✅ "Door ahead, 8 feet. Walk straight."

- ❌ "A person is detected"  
- ✅ "Person ahead, stepping aside. Wait 2 seconds."

Use this format:
- ALERT: Short, clear danger warning (2-6 words)
- GUIDANCE: "Chair left, go right." "Door ahead, 6 feet." (under 10 words)
- INFO: Slightly more detail if  needed

## OUTPUT FORMAT

Start EVERY response with "SPEAK: " or "SILENT: " (required).

SPEAK examples:
- "SPEAK: Chair left. Keep right."
- "SPEAK: Clear path. Door at end of hall."
- "SPEAK: Person approaching from right. Hold position."

SILENT example:
- "SILENT: Same hallway, clear path, no changes."

## NAVIGATION MODE (when navigation state is active)
- Use a structured loop: DISCOVER → if goal not found, CHANGE STATE (turn/move) → DISCOVER.
- If goal is found, give a short plan with spatial cues.
- While moving, keep giving brief safety and environment updates.

## KEY MINDSET

You are a GUIDE, not a narrator. The user is walking and needs real-time help:
- Short beats long
- Actions beat descriptions  
- Safety beats politeness
- Speaking beats missing an obstacle

If in doubt, SPEAK. A false alert is better than a collision."""


CONVERSATION_SYSTEM_PROMPT = """You are Helios, a helpful vision assistant for a blind person.

The user has asked you a question. You MUST answer it.

You receive:
1. Camera image showing what's in front of the user
2. YOLO object detection data (summary, objects with positions/distances)
3. Spatial context from recent observations (objects seen in the past 30 seconds)
4. The user's question

CRITICAL RULES:
- ALWAYS respond to the user's question - NEVER stay silent
- Be concise (under 30 words unless the question requires detail)
- Be direct and helpful
- Use spatial language: "left", "right", "ahead", "behind", "close", "far"
- If you can't answer from the image, say so briefly

## NAVIGATION MODE (when the user asks to go somewhere)
- Enter and persist navigation mode when the user asks to go/find a destination.
- Use a structured loop:
  1) DISCOVER: scan and describe what you can see related to the goal
  2) If goal not found, CHANGE STATE: ask the user to turn or move to a new view
  3) Repeat DISCOVER until the goal is found
- Once the goal is found, create a plan (micro-steps if the path is only partially visible).
- While executing a plan, keep giving short safety cues and environment updates.
- Keep responses concise, but allow extra words when giving step-by-step guidance.

## NAVIGATION STATE (metadata output)
You may receive a navigation state in the prompt with the current objective/phase/plan.
If navigation is active or requested, append a metadata block AFTER your spoken response:
<nav_state>{"active":true,"objective":"...","phase":"explore|plan|execute|complete","plan":["step 1","step 2"],"note":"short update"}</nav_state>
- Do NOT speak the <nav_state> block.
- Keep plan short (1-4 steps). Clear plan to [] once that step is done or you need a new plan.
- Set active=false and phase="complete" when the user reaches the goal or cancels.

OUTPUT FORMAT:
- DO NOT use any prefix
- Just provide your answer directly
- Use present tense
- Be conversational and friendly

Examples:
User: "What's in front of me?"
You: "A brown chair about 5 feet ahead, slightly to your left."

User: "Where can I sit?"
You: "There's a chair directly ahead, about 6 feet away."

User: "What does this say?"
You: "I can see text but it's too blurry to read clearly."

User: "Is there a person here?"
You: "No, I don't see any people in the current view."

Remember: You are answering a direct question from the user. Always provide a helpful response."""


@dataclass
class StreamedResponse:
    """Result from streaming Gemini call"""
    should_speak: bool
    full_text: str
    latency_ms: float
    total_ms: float


@dataclass
class NavigationState:
    """Persistent navigation state for structured guidance."""
    active: bool = False
    objective: Optional[str] = None
    phase: str = "idle"  # idle | explore | plan | execute | complete
    plan: List[str] = field(default_factory=list)
    last_update_ts: float = 0.0


class GeminiContextualNarrator:
    """
    Contextual Gemini narrator using standard API with streaming.
    Maintains conversation history and makes smart decisions about when to speak.
    """
    _NAV_STATE_PATTERN = re.compile(r"<nav_state>(.*?)</nav_state>", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        max_context_messages: int = 20,
        system_prompt: str = VISION_SYSTEM_PROMPT
    ):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        # Gemini 3 models require global endpoint
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

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
        self.system_prompt = system_prompt

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
        recent_history = scene_analysis.get("recent_history", None)
        urgency_level = scene_analysis.get("urgency_level", None)
        urgency_reason = scene_analysis.get("urgency_reason", None)
        motion = scene_analysis.get("motion", None)

        parts = []

        # Urgency level (for vision mode with heuristics)
        if urgency_level:
            parts.append(f"⚡ URGENCY: {urgency_level}")
            if urgency_reason:
                parts.append(f"   Reason: {urgency_reason}")
            parts.append("")

        # Recent history (for vision model context)
        if recent_history:
            parts.append(f"📜 RECENT HISTORY:\n{recent_history}\n")

        # Motion/sensor telemetry
        if motion:
            motion_fields = []
            for key in (
                "speed_mps",
                "speed_avg_1s_mps",
                "velocity_x_mps",
                "velocity_z_mps",
                "magnetic_x_ut",
                "magnetic_z_ut",
                "steps_last_3s",
                "steps_since_open",
                "is_moving",
            ):
                if key in motion:
                    value = motion.get(key)
                    if value is not None:
                        motion_fields.append(f"{key}={value}")
            if motion_fields:
                parts.append(f"📟 MOTION: {', '.join(motion_fields)}")

        # Navigation state (if active)
        navigation_state = scene_analysis.get("navigation_state")
        if navigation_state and navigation_state.get("active"):
            parts.append("🧭 NAVIGATION MODE: ACTIVE")
            objective = navigation_state.get("objective")
            phase = navigation_state.get("phase")
            plan = navigation_state.get("plan") or []
            plan_age = navigation_state.get("last_update_s_ago")

            if objective:
                parts.append(f"   Objective: {objective}")
            if phase:
                parts.append(f"   Phase: {phase}")
            if plan:
                parts.append(f"   Plan: {' | '.join(plan)}")
            if plan_age is not None:
                parts.append(f"   Plan age: {plan_age}s")

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
            parts.append("\n👤 User is walking (vision monitoring mode)")

        return "\n".join(parts)

    def _add_to_history(self, role: str, content: str):
        """Add message to history and trim if needed."""
        self.context_history.append({"role": role, "content": self._strip_nav_state_block(content)})

        # Trim old messages if exceeding limit
        if len(self.context_history) > self.max_context_messages:
            # Keep most recent messages
            self.context_history = self.context_history[-self.max_context_messages:]

    def _strip_nav_state_block(self, content: str) -> str:
        """Remove navigation state metadata blocks from stored history."""
        if not content:
            return content
        return self._NAV_STATE_PATTERN.sub("", content).strip()

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
        # Note: Using Gemini 3 Flash with minimal thinking for faster responses
        # Minimal thinking reduces token usage by ~30% while maintaining quality
        config = GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.1,
            max_output_tokens=2048,
            thinking_config=ThinkingConfig(thinking_level="minimal")  # Gemini 3: minimal thinking
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=messages,
            config=config
        )

        first_chunk_time = None
        full_response = ""

        import logging
        logger = logging.getLogger(__name__)

        try:
            chunk_count = 0
            logger.info(f"🔄 Starting to consume Gemini response stream...")
            async for chunk in response_stream:
                chunk_count += 1

                if chunk.text:
                    logger.info(f"📦 Chunk #{chunk_count}: '{chunk.text}' (len={len(chunk.text)})")

                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        logger.info(f"⏱️  First chunk received after {(first_chunk_time - start_time)*1000:.0f}ms")

                    full_response += chunk.text

                    # Heuristics already decided to speak, so always yield
                    # (Vision mode: heuristics triggered, Conversation mode: always responds)
                    yield (True, chunk.text)
                else:
                    # Empty chunk - check finish reason
                    logger.warning(f"⚠️ Empty chunk #{chunk_count}")
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'finish_reason'):
                                logger.warning(f"⚠️ Finish reason: {candidate.finish_reason}")

            logger.info(f"✅ Stream completed. Total chunks: {chunk_count} | Final response: '{full_response}'")
        except Exception as e:
            logger.error(f"❌ Stream error after {chunk_count} chunks: {e}", exc_info=True)
            raise

        end_time = time.perf_counter()

        # Debug logging - show raw Gemini response
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 RAW GEMINI | Full response with prefix: '{full_response}'")

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

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 GEMINI RESPONSE | Speak: {should_speak} | Text: '{full_text.strip()}' | Length: {len(full_text.strip())}")

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
            model="gemini-3-flash-preview",
            max_context_messages=20,  # Internal vision model context
            system_prompt=VISION_SYSTEM_PROMPT
        )

        # Conversation narrator - handles user queries (ALWAYS responds)
        self.conversation_narrator = GeminiContextualNarrator(
            model="gemini-3-flash-preview",
            max_context_messages=10,  # Recent conversation history
            system_prompt=CONVERSATION_SYSTEM_PROMPT  # Different prompt - no SPEAK/SILENT logic
        )

        # Spatial memory: rolling buffer of scene snapshots
        self.scene_history: deque[SceneSnapshot] = deque(
            maxlen=self.config.max_scene_history
        )

        # Cache latest frame for conversation queries
        self.latest_frame: Optional[str] = None
        self.latest_yolo: Optional[Dict[str, Any]] = None

        # Heuristics state tracking
        self.last_spoke_time: float = 0.0  # Timestamp of last speech output
        self.heuristics_config = HeuristicsConfig()

        # Persistent navigation state
        self.navigation_state = NavigationState()

    _NAV_STATE_PATTERN = re.compile(r"<nav_state>(.*?)</nav_state>", re.IGNORECASE | re.DOTALL)

    def _reset_navigation_state(self):
        """Clear navigation state."""
        self.navigation_state.active = False
        self.navigation_state.objective = None
        self.navigation_state.phase = "idle"
        self.navigation_state.plan = []
        self.navigation_state.last_update_ts = time.time()

    def _start_navigation(self, objective: str):
        """Start or update navigation with a new objective."""
        self.navigation_state.active = True
        self.navigation_state.objective = objective
        self.navigation_state.phase = "explore"
        self.navigation_state.plan = []
        self.navigation_state.last_update_ts = time.time()

    def _is_navigation_cancel(self, user_question: str) -> bool:
        """Check if the user is cancelling navigation."""
        text = user_question.lower()
        cancel_phrases = [
            "cancel navigation",
            "stop navigation",
            "end navigation",
            "cancel that",
            "never mind",
            "nevermind",
            "forget it",
            "stop guiding",
            "stop guidance",
        ]
        return any(phrase in text for phrase in cancel_phrases)

    def _is_navigation_request(self, user_question: str) -> bool:
        """Heuristic check for navigation intent."""
        text = user_question.lower()
        patterns = [
            r"\b(navigate|navigation)\b",
            r"\b(go|get|head|walk|move|lead|take)\s+(me\s+)?to\b",
            r"\bfind\b",
            r"\blocate\b",
            r"\blook for\b",
            r"\bwhere\s+is\b",
            r"\bwhere's\b",
            r"\bexit\b",
            r"\bleave\b",
            r"\bget out\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _extract_navigation_objective(self, user_question: str) -> str:
        """Extract a concise navigation target from the user's request."""
        text = user_question.strip()
        patterns = [
            r"\b(?:go|get|head|walk|move|lead|take)\s+(?:me\s+)?to\s+(.+)",
            r"\b(?:navigate|navigation)\s+to\s+(.+)",
            r"\b(?:find|locate|look for)\s+(.+)",
            r"\bwhere\s+is\s+(.+)",
            r"\bwhere's\s+(.+)",
            r"\bexit\s+(.+)",
            r"\bleave\s+(.+)",
            r"\bget out(?: of)?\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                objective = match.group(1)
                return objective.strip(" .,!?:;\"'").strip()
        return text.strip(" .,!?:;\"'").strip()

    def _handle_navigation_intent(self, user_question: str):
        """Update navigation state based on the current user request."""
        if not user_question:
            return
        if self._is_navigation_cancel(user_question):
            self._reset_navigation_state()
            return
        if self._is_navigation_request(user_question):
            objective = self._extract_navigation_objective(user_question)
            if (not self.navigation_state.active) or (objective != self.navigation_state.objective):
                self._start_navigation(objective)
            else:
                self.navigation_state.last_update_ts = time.time()

    def _navigation_prompt_payload(self) -> Optional[Dict[str, Any]]:
        """Build navigation state payload for prompt injection."""
        # TODO: consider programmatic cue cadence (e.g., every 3s or 3 steps) if needed.
        if not self.navigation_state.active:
            return None
        payload: Dict[str, Any] = {
            "active": True,
            "objective": self.navigation_state.objective,
            "phase": self.navigation_state.phase,
            "plan": self.navigation_state.plan[:4],
        }
        if self.navigation_state.last_update_ts:
            payload["last_update_s_ago"] = int(time.time() - self.navigation_state.last_update_ts)
        return payload

    def _strip_code_fence(self, text: str) -> str:
        """Strip surrounding code fences if present."""
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return text

    def _extract_nav_state_block(self, text: str) -> tuple[str, Optional[dict]]:
        """Extract navigation state metadata from the model response."""
        if not text:
            return text, None
        match = self._NAV_STATE_PATTERN.search(text)
        if not match:
            return text.strip(), None
        json_blob = self._strip_code_fence(match.group(1).strip())
        cleaned = (text[:match.start()] + text[match.end():]).strip()
        try:
            nav_update = json.loads(json_blob)
        except json.JSONDecodeError:
            nav_update = None
        return cleaned, nav_update

    def _apply_navigation_update(self, nav_update: dict):
        """Apply a navigation state update from the model."""
        if not isinstance(nav_update, dict):
            return

        active = nav_update.get("active")
        phase = nav_update.get("phase")

        if active is False or phase == "complete":
            self._reset_navigation_state()
            self.navigation_state.phase = "complete"
            return

        if active is True:
            self.navigation_state.active = True

        objective = nav_update.get("objective")
        if isinstance(objective, str) and objective.strip():
            self.navigation_state.objective = objective.strip()

        if isinstance(phase, str) and phase.strip():
            self.navigation_state.phase = phase.strip()

        if nav_update.get("clear_plan") is True:
            self.navigation_state.plan = []
        else:
            plan = nav_update.get("plan")
            if isinstance(plan, str) and plan.strip():
                self.navigation_state.plan = [plan.strip()]
            elif isinstance(plan, list):
                cleaned_plan = [str(step).strip() for step in plan if str(step).strip()]
                self.navigation_state.plan = cleaned_plan[:4]

        self.navigation_state.last_update_ts = time.time()

    async def process_frame(
        self,
        frame_base64: str,
        yolo_objects: Dict[str, Any]
    ) -> Optional[str]:
        """
        Vision Pipeline: Process incoming frame (called every ~1 second).

        Uses heuristics engine to decide when to call Gemini for proactive guidance.
        Gemini is only called when heuristics determine something needs to be said.

        Args:
            frame_base64: Base64-encoded frame image
            yolo_objects: YOLO detection results with summary, objects, emergency flag

        Returns:
            Text to speak if heuristics trigger, None otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        # Update cache for conversation queries
        self.latest_frame = frame_base64
        self.latest_yolo = yolo_objects

        # Calculate time since last speech for debouncing
        current_time = time.time()
        last_spoke_seconds_ago = current_time - self.last_spoke_time

        # Get recent object labels for debouncing INFO-level alerts
        recent_objects = self._get_recent_object_labels(lookback_seconds=10.0)

        # Use heuristics to decide if we should speak
        decision: SpeakDecision = evaluate_scene(
            scene_analysis=yolo_objects,
            recent_objects=recent_objects,
            last_spoke_seconds_ago=last_spoke_seconds_ago,
            config=self.heuristics_config
        )

        logger.info(
            f"🎯 HEURISTICS | should_speak={decision.should_speak} | "
            f"urgency={decision.urgency.value} | reason={decision.reason}"
        )

        # Always update spatial memory
        snapshot = SceneSnapshot(
            timestamp=current_time,
            yolo_objects=yolo_objects,
            scene_description=decision.reason,
            frame_base64=frame_base64 if self.config.store_frames else None
        )
        self.scene_history.append(snapshot)

        # If heuristics say don't speak, stay silent (no Gemini call)
        if not decision.should_speak:
            return None

        # Heuristics triggered: call Gemini with urgency context
        logger.info(f"🔊 VISION SPEAKING | Urgency: {decision.urgency.value}")

        # Build scene analysis with urgency context for Gemini
        scene_with_urgency = {
            **yolo_objects,
            "urgency_level": decision.urgency.value.upper(),
            "urgency_reason": decision.reason,
            "recent_history": self._build_vision_history_context()
        }
        navigation_payload = self._navigation_prompt_payload()
        if navigation_payload:
            scene_with_urgency["navigation_state"] = navigation_payload

        # Call Gemini vision narrator
        response = await self.vision_narrator.process_input(
            scene_analysis=scene_with_urgency,
            image_base64=frame_base64,
            user_question=None  # Vision mode, not conversation
        )

        # Update last spoke time
        if response.full_text:
            self.last_spoke_time = current_time
            # Update the snapshot with actual Gemini response
            self.scene_history[-1] = SceneSnapshot(
                timestamp=current_time,
                yolo_objects=yolo_objects,
                scene_description=response.full_text,
                frame_base64=frame_base64 if self.config.store_frames else None
            )

        return response.full_text if response.full_text else None

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
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"💬 CONVERSATION PIPELINE | Question: '{user_question}'")

        # Update navigation state from user intent (start/stop)
        self._handle_navigation_intent(user_question)

        if not self.latest_frame:
            logger.warning("⚠️ No latest frame available yet")
            return "I haven't seen anything yet. Please wait a moment."

        logger.info(f"✅ Latest frame available | Has YOLO: {self.latest_yolo is not None}")

        # Build enriched spatial context from vision history
        logger.info("🔨 Building spatial context from vision history...")
        spatial_context = self._build_spatial_context()
        logger.info(f"📝 Spatial context built: {len(spatial_context)} chars")

        # Create scene analysis with spatial context
        scene_analysis = {
            "summary": spatial_context,
            "objects": self.latest_yolo.get("objects", []) if self.latest_yolo else [],
            "emergency_stop": False  # User questions aren't emergencies
        }
        navigation_payload = self._navigation_prompt_payload()
        if navigation_payload:
            scene_analysis["navigation_state"] = navigation_payload
        logger.info(f"📊 Scene analysis | Objects: {len(scene_analysis['objects'])}")

        # Conversation model gets vision context + question
        logger.info("🤖 Calling conversation narrator.process_input...")
        response = await self.conversation_narrator.process_input(
            scene_analysis=scene_analysis,
            image_base64=self.latest_frame,
            user_question=user_question
        )

        logger.info(f"✅ Got response | Should speak: {response.should_speak} | Text: '{response.full_text}' | Length: {len(response.full_text)}")

        spoken_text, nav_update = self._extract_nav_state_block(response.full_text)
        if nav_update:
            self._apply_navigation_update(nav_update)
        return spoken_text

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
            return f"Previous: {self._summarize_scene(recent_scenes[0])}"

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

            desc = self._summarize_scene(scene)
            history_items.append(f"[{time_str}] {desc}")

        return "\n".join(history_items)

    def _summarize_scene(self, scene: SceneSnapshot) -> str:
        """
        Build a useful summary from a scene snapshot.

        For frames where Gemini spoke, uses the actual response.
        For silent frames, builds a summary from YOLO data.
        """
        desc = scene.scene_description

        # If we have a real Gemini response (not a heuristics reason), use it
        if desc and not desc.startswith("Clear path") and not desc.startswith("In path:") \
                and not desc.startswith("Immediate:") and not desc.startswith("New:") \
                and not desc.startswith("Emergency"):
            # This is likely an actual Gemini response
            if desc.startswith("SPEAK: "):
                return desc[7:].strip()
            return desc

        # For silent frames or heuristic-only frames, build from YOLO data
        objects = scene.yolo_objects.get("objects", [])
        if not objects:
            return "Clear path"

        # Build compact object summary
        obj_summaries = []
        for obj in objects[:4]:  # Max 4 objects
            label = obj.get("label", "object")
            pos = obj.get("position", "")
            dist = obj.get("distance", "")

            if pos and dist:
                obj_summaries.append(f"{label} ({pos}, {dist})")
            elif pos:
                obj_summaries.append(f"{label} ({pos})")
            else:
                obj_summaries.append(label)

        return ", ".join(obj_summaries) if obj_summaries else "Clear path"

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

    def _get_recent_object_labels(self, lookback_seconds: float = 10.0) -> set[str]:
        """
        Get labels of objects seen in the recent past for debouncing.

        Args:
            lookback_seconds: How far back to look (default 10 seconds)

        Returns:
            Set of object labels seen recently
        """
        if not self.scene_history:
            return set()

        cutoff_time = time.time() - lookback_seconds
        recent_labels: set[str] = set()

        for scene in self.scene_history:
            if scene.timestamp >= cutoff_time:
                for obj in scene.yolo_objects.get("objects", []):
                    label = obj.get("label")
                    if label:
                        recent_labels.add(label)

        return recent_labels

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
