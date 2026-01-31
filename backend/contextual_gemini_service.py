"""
Contextual Gemini Service with Streaming Text Output
Uses standard Gemini API (generate_content_stream) for decision-making and text streaming.
"""
import base64
import os
import time
from dataclasses import dataclass
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, GenerateContentConfig

load_dotenv()

SYSTEM_PROMPT = """You are a real-time navigation assistant for a blind person wearing a camera.

You receive:
1. Camera image
2. YOLO object detection data (summary, objects with positions/distances)
3. Optional user question

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
        model: str = "gemini-2.0-flash",
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

        parts = []

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
