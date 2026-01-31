"""
Gemini Narration Service for Blind User Assistance

Low-latency, cached narration generation from YOLO scene analysis.
"""

import os
import hashlib
from typing import Generator
from google import genai
from dotenv import load_dotenv
from cachetools import TTLCache

load_dotenv()

SYSTEM_PROMPT = """You are a concise audio guide for a blind person wearing a camera.
Given a scene description, provide a brief, helpful narration.
- Prioritize safety (vehicles, obstacles)
- Be direct and spatial ("on your left", "ahead")
- Keep responses under 20 words unless urgent
- Use present tense"""


class GeminiNarrator:
    def __init__(self, model: str = "gemini-2.0-flash", cache_ttl: int = 30, cache_maxsize: int = 100):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)

    def _cache_key(self, scene_analysis: dict) -> str:
        """Generate cache key from semantic content (ignores timestamps/boxes)."""
        summary = scene_analysis.get("summary", "")
        objects = scene_analysis.get("objects", [])
        labels = sorted(obj.get("label", "") for obj in objects)

        key_str = f"{summary}_{'_'.join(labels)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _build_prompt(self, scene_analysis: dict) -> str:
        """Build the prompt for Gemini from scene analysis."""
        summary = scene_analysis.get("summary", "Nothing detected")
        emergency = scene_analysis.get("emergency_stop", False)

        prompt = f"Scene: {summary}"
        if emergency:
            prompt = f"URGENT - Vehicle nearby! {prompt}"

        return prompt

    def narrate(self, scene_analysis: dict) -> str:
        """Get narration for scene (blocking, uses cache)."""
        cache_key = self._cache_key(scene_analysis)

        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = self._build_prompt(scene_analysis)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
                {"role": "model", "parts": [{"text": "I understand. I'll provide brief, spatial guidance for a blind person."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        )

        result = response.text
        self.cache[cache_key] = result
        return result

    def narrate_stream(self, scene_analysis: dict) -> Generator[str, None, None]:
        """Stream narration for scene (yields chunks, caches final result)."""
        cache_key = self._cache_key(scene_analysis)

        if cache_key in self.cache:
            yield self.cache[cache_key]
            return

        prompt = self._build_prompt(scene_analysis)

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
                {"role": "model", "parts": [{"text": "I understand. I'll provide brief, spatial guidance for a blind person."}]},
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        )

        full_response = []
        for chunk in response:
            text = chunk.text
            full_response.append(text)
            yield text

        self.cache[cache_key] = "".join(full_response)
