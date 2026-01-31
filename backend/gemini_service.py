"""
Gemini Live API Narration Service for Blind User Assistance

Low-latency, real-time audio narration using Vertex AI Live API.
Combines YOLO scene analysis with raw image for rich understanding.
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import io
import os
import time
import wave
from collections import Counter, deque
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from cachetools import TTLCache
from dotenv import load_dotenv
from google import genai
from google.genai.types import (LiveConnectConfig, Part, SpeechConfig,
                                VoiceConfig)

load_dotenv()


class RateLimiter:
    """Simple sliding window rate limiter."""

    def __init__(self, max_calls: int = 10, window_seconds: float = 60.0):
        """
        Args:
            max_calls: Maximum calls allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = deque()

    def is_allowed(self) -> bool:
        """Check if a call is allowed and record it if so."""
        now = time.time()

        # Remove old calls outside the window
        while self.calls and self.calls[0] < now - self.window_seconds:
            self.calls.popleft()

        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

    def wait_time(self) -> float:
        """Return seconds until next call is allowed (0 if allowed now)."""
        if len(self.calls) < self.max_calls:
            return 0.0
        oldest = self.calls[0]
        return max(0.0, oldest + self.window_seconds - time.time())

# Set the credentials path for Google Cloud
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    # Handle relative paths
    if not os.path.isabs(credentials_path):
        credentials_path = os.path.join(os.path.dirname(__file__), credentials_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

SYSTEM_PROMPT = """You are a concise audio guide who used to be a pirate, for a blind person wearing a camera.
You receive both an image and object detection data. Use both to give helpful guidance.
- Prioritize safety (vehicles, obstacles, stairs, curbs) ARGGGGGGGGHHH
- Be direct and spatial ("on your left", "three steps ahead")
- Keep responses under 25 words unless urgent
- Use present tense
- If emergency_stop is true, start with a WARNING
- Say exact strings like "ARGHHHOOOOOEEWOOOOUUHHHHGGHHHRRERRRRR" and "AHOYYYYYYYYYYY MATEYYYYYYYYYY" occasionally to maintain pirate persona"""


@dataclass
class NarrationResult:
    """Result from narration with timing metrics."""
    audio_data: bytes  # Raw PCM audio (24kHz, 16-bit, mono)
    transcript: str    # Text transcript if available
    latency_ms: float  # Time to first audio chunk
    total_ms: float    # Total processing time
    cached: bool       # Whether result was from cache

    def save_wav(self, filepath: str):
        """Save audio as WAV file."""
        with wave.open(filepath, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(24000)  # 24kHz
            wav.writeframes(self.audio_data)


class GeminiLiveNarrator:
    """
    Real-time narrator using Gemini Live API via Vertex AI.
    Outputs audio narration for blind user assistance.
    """

    def __init__(
        self,
        model: str = "gemini-live-2.5-flash-native-audio",
        cache_ttl: int = 30,
        cache_maxsize: int = 100,
        rate_limit_calls: int = 10,
        rate_limit_window: float = 60.0
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
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, window_seconds=rate_limit_window)

    def _cache_key(self, scene_analysis: dict, image_hash: Optional[str] = None) -> str:
        """Generate cache key from semantic content."""
        summary = scene_analysis.get("summary", "")
        objects = scene_analysis.get("objects", [])
        labels = sorted(obj.get("label", "") for obj in objects)

        key_str = f"{summary}_{'_'.join(labels)}"
        if image_hash:
            key_str += f"_{image_hash[:16]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _build_prompt(self, scene_analysis: dict) -> str:
        """Build text prompt from YOLO scene analysis."""
        summary = scene_analysis.get("summary", "Nothing detected")
        emergency = scene_analysis.get("emergency_stop", False)
        objects = scene_analysis.get("objects", [])

        parts = []
        if emergency:
            parts.append("EMERGENCY: Vehicle very close!")

        parts.append(f"Detected: {summary}")

        if objects:
            details = []
            for obj in objects[:5]:
                label = obj.get("label", "object")
                pos = obj.get("position", "unknown")
                dist = obj.get("distance", "unknown")
                details.append(f"{label} ({pos}, {dist})")
            parts.append(f"Details: {', '.join(details)}")

        return " | ".join(parts)

    def _decode_image(self, image_data: str) -> bytes:
        """Decode base64 image, handling data URL prefix."""
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)

    async def narrate_with_image(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> NarrationResult:
        """
        Get audio narration using both YOLO data and raw image.

        Args:
            scene_analysis: YOLO output with summary, objects, emergency_stop
            image_base64: Base64-encoded JPEG from camera

        Returns:
            NarrationResult with audio data and timing metrics
        """
        start_time = time.perf_counter()

        # Check cache first (cache hits don't count against rate limit)
        image_bytes = self._decode_image(image_base64)
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cache_key = self._cache_key(scene_analysis, image_hash)

        if cache_key in self.cache:
            elapsed = (time.perf_counter() - start_time) * 1000
            cached_result = self.cache[cache_key]
            return NarrationResult(
                audio_data=cached_result["audio"],
                transcript=cached_result.get("transcript", ""),
                latency_ms=elapsed,
                total_ms=elapsed,
                cached=True
            )

        # Rate limit check (only for non-cached requests)
        if not self.rate_limiter.is_allowed():
            wait = self.rate_limiter.wait_time()
            raise RuntimeError(f"RATE LIMITED SLOW DOWN OMG THIS EXPENSIVE. Try again in {wait:.1f}s")

        # Build multimodal content
        text_prompt = self._build_prompt(scene_analysis)
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Configure for audio output
        config = LiveConnectConfig(
            system_instruction=SYSTEM_PROMPT,
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config={"voice_name": "Puck"}  # Clear, friendly voice
                )
            )
        )

        first_audio_time = None
        audio_chunks = []
        transcript = ""

        async with self.client.aio.live.connect(
            model=self.model,
            config=config
        ) as session:
            # Send image and text together
            await session.send_client_content(
                turns=[
                    {"role": "user", "parts": [image_part, {"text": text_prompt}]}
                ],
                turn_complete=True
            )

            # Collect audio response
            async for response in session.receive():
                # Check for audio data
                if response.data:
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                    audio_chunks.append(response.data)

                # Check for transcript (if available)
                if hasattr(response, 'server_content') and response.server_content:
                    if hasattr(response.server_content, 'output_transcription'):
                        transcript = response.server_content.output_transcription or ""

        end_time = time.perf_counter()
        audio_data = b"".join(audio_chunks)

        # Cache the result
        self.cache[cache_key] = {"audio": audio_data, "transcript": transcript}

        latency = ((first_audio_time or end_time) - start_time) * 1000
        total = (end_time - start_time) * 1000

        return NarrationResult(
            audio_data=audio_data,
            transcript=transcript,
            latency_ms=latency,
            total_ms=total,
            cached=False
        )

    async def narrate_with_image_stream(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> AsyncGenerator[tuple[bytes, float], None]:
        """
        Stream audio chunks with timing.

        Yields:
            Tuples of (audio_chunk_bytes, elapsed_ms)
        """
        start_time = time.perf_counter()

        image_bytes = self._decode_image(image_base64)
        text_prompt = self._build_prompt(scene_analysis)
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        config = LiveConnectConfig(
            system_instruction=SYSTEM_PROMPT,
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config={"voice_name": "Puck"}
                )
            )
        )

        async with self.client.aio.live.connect(
            model=self.model,
            config=config
        ) as session:
            await session.send_client_content(
                turns=[
                    {"role": "user", "parts": [image_part, {"text": text_prompt}]}
                ],
                turn_complete=True
            )

            async for response in session.receive():
                if response.data:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    yield (response.data, elapsed)


class GeminiLiveSession:
    """
    Persistent Gemini Live session with continuous context updates.

    This maintains a single WebSocket connection and feeds scene updates
    continuously, only requesting audio responses when needed.
    """

    def __init__(
        self,
        model: str = "gemini-live-2.5-flash-native-audio",
        min_narration_interval: float = 5.0,
        rate_limit_calls: int = 20,
        rate_limit_window: float = 60.0
    ):
        """
        Args:
            model: Gemini model name
            min_narration_interval: Minimum seconds between auto-narrations
            rate_limit_calls: Max API responses per window
            rate_limit_window: Rate limit window in seconds
        """
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
        self.config = LiveConnectConfig(
            system_instruction=SYSTEM_PROMPT,
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config={"voice_name": "Puck"}
                )
            )
        )

        # Session state
        self.session = None
        self.is_connected = False
        self._connect_lock = asyncio.Lock()

        # Narration control
        self.min_narration_interval = min_narration_interval
        self.last_narration_time = 0
        self.last_scene: Optional[dict] = None  # Store last scene for similarity comparison
        self.similarity_threshold = 0.7  # Below this = scene changed enough to narrate
        self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, window_seconds=rate_limit_window)

        # Context tracking
        self.frames_since_narration = 0
        self.current_emergency = False

    async def connect(self):
        """Open persistent WebSocket session."""
        async with self._connect_lock:
            if self.is_connected:
                return

            self.session = await self.client.aio.live.connect(
                model=self.model,
                config=self.config
            ).__aenter__()
            self.is_connected = True

    async def disconnect(self):
        """Close the session."""
        async with self._connect_lock:
            if self.session and self.is_connected:
                await self.session.__aexit__(None, None, None)
                self.session = None
                self.is_connected = False

    async def ensure_connected(self):
        """Reconnect if needed."""
        if not self.is_connected:
            await self.connect()

    def _decode_image(self, image_data: str) -> bytes:
        """Decode base64 image, handling data URL prefix."""
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)

    def _build_prompt(self, scene_analysis: dict) -> str:
        """Build text prompt from YOLO scene analysis."""
        summary = scene_analysis.get("summary", "Nothing detected")
        emergency = scene_analysis.get("emergency_stop", False)
        objects = scene_analysis.get("objects", [])

        parts = []
        if emergency:
            parts.append("EMERGENCY: Vehicle very close!")

        parts.append(f"Scene: {summary}")

        if objects:
            details = []
            for obj in objects[:5]:
                label = obj.get("label", "object")
                pos = obj.get("position", "unknown")
                dist = obj.get("distance", "unknown")
                details.append(f"{label} ({pos}, {dist})")
            parts.append(f"Details: {', '.join(details)}")

        return " | ".join(parts)

    def _scene_similarity(self, old_scene: Optional[dict], new_scene: dict) -> float:
        """
        Compare scenes semantically using object labels, counts, and distances.

        Returns:
            float: 0.0 to 1.0 where 1.0 = identical scenes
        """
        if old_scene is None:
            return 0.0  # No previous scene, definitely narrate

        old_objects = old_scene.get("objects", [])
        new_objects = new_scene.get("objects", [])

        # Count objects by label
        old_labels = Counter(obj.get("label") for obj in old_objects)
        new_labels = Counter(obj.get("label") for obj in new_objects)

        # Count objects by (label, distance) - more sensitive to proximity changes
        old_label_dist = Counter((obj.get("label"), obj.get("distance")) for obj in old_objects)
        new_label_dist = Counter((obj.get("label"), obj.get("distance")) for obj in new_objects)

        # Check for new object types (important for navigation)
        new_types = set(new_labels.keys()) - set(old_labels.keys())
        if new_types:
            # New object type appeared - likely important
            return 0.0

        # Check for distance changes (far->close is important)
        for obj in new_objects:
            label = obj.get("label")
            dist = obj.get("distance")
            if dist in ("immediate", "close"):
                # Check if this was previously far or not present
                old_close = sum(1 for o in old_objects
                               if o.get("label") == label and o.get("distance") in ("immediate", "close"))
                new_close = sum(1 for o in new_objects
                               if o.get("label") == label and o.get("distance") in ("immediate", "close"))
                if new_close > old_close:
                    return 0.3  # Something got closer - moderately important

        # Calculate label count similarity
        all_labels = set(old_labels.keys()) | set(new_labels.keys())
        if not all_labels:
            return 1.0  # Both empty

        label_diff = sum(abs(old_labels.get(l, 0) - new_labels.get(l, 0)) for l in all_labels)
        label_total = sum(old_labels.values()) + sum(new_labels.values())

        if label_total == 0:
            return 1.0

        label_similarity = 1.0 - (label_diff / label_total)

        # Calculate label+distance similarity (more granular)
        all_label_dist = set(old_label_dist.keys()) | set(new_label_dist.keys())
        dist_diff = sum(abs(old_label_dist.get(ld, 0) - new_label_dist.get(ld, 0)) for ld in all_label_dist)
        dist_total = sum(old_label_dist.values()) + sum(new_label_dist.values())

        if dist_total == 0:
            dist_similarity = 1.0
        else:
            dist_similarity = 1.0 - (dist_diff / dist_total)

        # Weighted average: distance changes matter more
        return 0.4 * label_similarity + 0.6 * dist_similarity

    async def _llm_should_narrate(self, scene_analysis: dict, image_base64: str) -> tuple[bool, str]:
        """
        Ask the LLM if the scene has changed enough to warrant narration.

        NOTE: Not currently used. This is a placeholder for future use.
        Uses slightly more API cost but makes smarter decisions.

        Returns:
            (should_narrate, reason) tuple
        """
        # TODO: Implement LLM-based decision making
        # Would send a text-only query like:
        # "Based on the previous context, has the scene changed enough to update the user?
        #  Reply YES or NO followed by a brief reason."
        # Then parse the response.
        #
        # For now, fall back to similarity-based approach
        similarity = self._scene_similarity(self.last_scene, scene_analysis)
        if similarity < self.similarity_threshold:
            return True, f"llm_decided_change (similarity={similarity:.2f})"
        return False, f"llm_decided_same (similarity={similarity:.2f})"

    def should_narrate(self, scene_analysis: dict) -> tuple[bool, str]:
        """
        Determine if we should request a narration now.
        Uses semantic similarity to detect meaningful scene changes.

        Returns:
            (should_narrate, reason) tuple
        """
        now = time.time()
        emergency = scene_analysis.get("emergency_stop", False)

        # Always narrate emergencies immediately
        if emergency and not self.current_emergency:
            self.current_emergency = True
            return True, "emergency"

        # Clear emergency flag when no longer emergency
        if not emergency:
            self.current_emergency = False

        # Check time throttle
        time_since_last = now - self.last_narration_time
        if time_since_last < self.min_narration_interval:
            return False, "throttled"

        # Check semantic similarity (not exact hash match)
        similarity = self._scene_similarity(self.last_scene, scene_analysis)

        if similarity >= self.similarity_threshold:
            return False, f"similar_scene ({similarity:.2f})"

        # Scene changed significantly and enough time passed
        return True, f"scene_changed ({similarity:.2f})"

    async def update_context(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> None:
        """
        Feed new scene data to maintain context WITHOUT requesting a response.
        This is cheap - just keeps the model informed.

        Args:
            scene_analysis: YOLO output with summary, objects, emergency_stop
            image_base64: Base64-encoded JPEG from camera
        """
        await self.ensure_connected()

        image_bytes = self._decode_image(image_base64)
        text_prompt = f"[Context Update] {self._build_prompt(scene_analysis)}"
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Send without requesting response (turn_complete=False)
        await self.session.send_client_content(
            turns=[{"role": "user", "parts": [image_part, {"text": text_prompt}]}],
            turn_complete=False  # Key: don't trigger response
        )

        self.frames_since_narration += 1

    async def request_narration(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> Optional[NarrationResult]:
        """
        Request audio narration of current scene.
        Call this on emergencies, significant changes, or intervals.

        Args:
            scene_analysis: YOLO output
            image_base64: Base64-encoded JPEG

        Returns:
            NarrationResult with audio, or None if rate limited
        """
        # Rate limit check
        if not self.rate_limiter.is_allowed():
            return None

        await self.ensure_connected()

        start_time = time.perf_counter()

        image_bytes = self._decode_image(image_base64)
        text_prompt = f"Describe what you see for navigation: {self._build_prompt(scene_analysis)}"
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Send and request response (turn_complete=True)
        await self.session.send_client_content(
            turns=[{"role": "user", "parts": [image_part, {"text": text_prompt}]}],
            turn_complete=True
        )

        # Collect audio response
        first_audio_time = None
        audio_chunks = []
        transcript = ""

        async for response in self.session.receive():
            if response.data:
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()
                audio_chunks.append(response.data)

            if hasattr(response, 'server_content') and response.server_content:
                if hasattr(response.server_content, 'output_transcription'):
                    transcript = response.server_content.output_transcription or ""
                # Check for turn complete
                if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                    break

        end_time = time.perf_counter()

        # Update tracking
        self.last_narration_time = time.time()
        self.last_scene = scene_analysis  # Store for similarity comparison
        self.frames_since_narration = 0

        return NarrationResult(
            audio_data=b"".join(audio_chunks),
            transcript=transcript,
            latency_ms=((first_audio_time or end_time) - start_time) * 1000,
            total_ms=(end_time - start_time) * 1000,
            cached=False
        )

    async def answer_question(
        self,
        question: str,
        scene_analysis: Optional[dict] = None,
        image_base64: Optional[str] = None
    ) -> Optional[NarrationResult]:
        """
        Answer a user's spoken question using current context.

        Args:
            question: The user's question (transcribed from audio)
            scene_analysis: Optional current scene (if available)
            image_base64: Optional current image (if available)

        Returns:
            NarrationResult with audio answer
        """
        if not self.rate_limiter.is_allowed():
            return None

        await self.ensure_connected()

        start_time = time.perf_counter()

        # Build the question prompt
        parts = []

        # Add image if provided
        if image_base64:
            image_bytes = self._decode_image(image_base64)
            parts.append(Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        # Add scene context if provided
        context = ""
        if scene_analysis:
            context = f"[Current scene: {self._build_prompt(scene_analysis)}] "

        parts.append({"text": f"{context}The user asks: {question}"})

        # Send question and request response
        await self.session.send_client_content(
            turns=[{"role": "user", "parts": parts}],
            turn_complete=True
        )

        # Collect audio response
        first_audio_time = None
        audio_chunks = []
        transcript = ""

        async for response in self.session.receive():
            if response.data:
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()
                audio_chunks.append(response.data)

            if hasattr(response, 'server_content') and response.server_content:
                if hasattr(response.server_content, 'output_transcription'):
                    transcript = response.server_content.output_transcription or ""
                if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                    break

        end_time = time.perf_counter()

        return NarrationResult(
            audio_data=b"".join(audio_chunks),
            transcript=transcript,
            latency_ms=((first_audio_time or end_time) - start_time) * 1000,
            total_ms=(end_time - start_time) * 1000,
            cached=False
        )

    async def process_frame(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> Optional[NarrationResult]:
        """
        Smart frame processing: updates context and narrates only when needed.

        This is the main entry point for processing frames. It:
        1. Always updates the model's context with the new scene
        2. Only requests audio narration when appropriate

        Args:
            scene_analysis: YOLO output
            image_base64: Base64-encoded JPEG

        Returns:
            NarrationResult if narration was triggered, None otherwise
        """
        # Always update context (cheap, no response)
        await self.update_context(scene_analysis, image_base64)

        # Check if we should narrate
        should_speak, reason = self.should_narrate(scene_analysis)

        if should_speak:
            return await self.request_narration(scene_analysis, image_base64)

        return None


# Convenience wrapper for sync usage
class GeminiNarrator:
    """Synchronous wrapper around GeminiLiveNarrator."""

    def __init__(self, model: str = "gemini-live-2.5-flash-native-audio"):
        self._async_narrator = GeminiLiveNarrator(model=model)

    def narrate_with_image(
        self,
        scene_analysis: dict,
        image_base64: str
    ) -> NarrationResult:
        """Sync version of narrate_with_image."""
        try:
            # Check if there's already a running event loop
            asyncio.get_running_loop()
            # If we get here, there's a running loop - run in a thread pool
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self._async_narrator.narrate_with_image(scene_analysis, image_base64)
                )
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run() directly
            return asyncio.run(
                self._async_narrator.narrate_with_image(scene_analysis, image_base64)
            )
