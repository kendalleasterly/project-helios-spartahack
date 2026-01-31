"""
Gemini Live API Narration Service for Blind User Assistance

Low-latency, real-time audio narration using Vertex AI Live API.
Combines YOLO scene analysis with raw image for rich understanding.
"""

import os
import time
import base64
import asyncio
import hashlib
import wave
import io
import concurrent.futures
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
from google import genai
from google.genai.types import LiveConnectConfig, Part, SpeechConfig, VoiceConfig
from dotenv import load_dotenv
from cachetools import TTLCache

load_dotenv()

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
        cache_maxsize: int = 100
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

        # Check cache
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
