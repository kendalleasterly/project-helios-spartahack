"""
Test script for GeminiLiveNarrator with audio output.

Run: python test_gemini_service.py [path_to_test_image.jpg]
"""

import sys
import asyncio
import base64
from pathlib import Path
from gemini_service import GeminiLiveNarrator, GeminiNarrator

# Mock scene_analysis data (matches YOLO server output format)
MOCK_SCENES = {
    "person_ahead": {
        "timestamp": "2026-01-31T12:00:01.000Z",
        "emergency_stop": False,
        "summary": "Person (mid-center, close)",
        "objects": [
            {"id": 0, "label": "person", "confidence": 0.95, "position": "mid-center", "distance": "close", "box": [200, 100, 400, 450]}
        ]
    },
    "crowded": {
        "timestamp": "2026-01-31T12:00:02.000Z",
        "emergency_stop": False,
        "summary": "A dense cluster of 3 people in the center, chair on the left",
        "objects": [
            {"id": 0, "label": "person", "confidence": 0.95, "position": "mid-center", "distance": "close", "box": [200, 100, 350, 450]},
            {"id": 1, "label": "person", "confidence": 0.92, "position": "mid-center", "distance": "close", "box": [280, 110, 420, 460]},
            {"id": 2, "label": "person", "confidence": 0.88, "position": "mid-center", "distance": "close", "box": [320, 90, 480, 440]},
            {"id": 3, "label": "chair", "confidence": 0.85, "position": "mid-left", "distance": "far", "box": [10, 200, 100, 350]}
        ]
    },
    "emergency": {
        "timestamp": "2026-01-31T12:00:03.000Z",
        "emergency_stop": True,
        "summary": "Car (mid-right, immediate)",
        "objects": [
            {"id": 0, "label": "car", "confidence": 0.97, "position": "mid-right", "distance": "immediate", "box": [400, 50, 640, 400]}
        ]
    }
}


def create_test_image() -> str:
    """Create a simple test image as base64."""
    try:
        from PIL import Image
        import io

        # Create a simple 640x480 test image
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    except ImportError:
        print("PIL not available, using minimal JPEG")
        # Minimal valid JPEG
        minimal_jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
            0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
            0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
            0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
            0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x7E, 0xCB,
            0xD3, 0x46, 0x10, 0x9B, 0xA6, 0x9E, 0x9F, 0xFF, 0xD9
        ])
        return base64.b64encode(minimal_jpeg).decode()


def load_image(path: str) -> str:
    """Load image from file as base64."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


async def test_async_narrator(image_base64: str):
    """Test async GeminiLiveNarrator with audio output."""
    print("=" * 60)
    print("TEST: Async GeminiLiveNarrator - Audio Output")
    print("=" * 60)

    narrator = GeminiLiveNarrator()

    for name, scene in MOCK_SCENES.items():
        print(f"\n--- Scene: {name} ---")
        print(f"YOLO: {scene['summary']}")
        if scene.get('emergency_stop'):
            print("⚠️  EMERGENCY FLAG SET")

        result = await narrator.narrate_with_image(scene, image_base64)

        status = "CACHED" if result.cached else "LIVE"
        audio_kb = len(result.audio_data) / 1024
        duration_sec = len(result.audio_data) / (24000 * 2)  # 24kHz, 16-bit

        print(f"\n[{status}] Latency: {result.latency_ms:.0f}ms | Total: {result.total_ms:.0f}ms")
        print(f"Audio: {audio_kb:.1f}KB ({duration_sec:.1f}s)")
        if result.transcript:
            print(f"Transcript: {result.transcript}")

        # Save first audio to file for testing
        if name == "person_ahead":
            result.save_wav("test_output.wav")
            print("Saved: test_output.wav")

        # Test cache hit
        result2 = await narrator.narrate_with_image(scene, image_base64)
        print(f"[{'CACHED' if result2.cached else 'LIVE'}] Cache check: {result2.latency_ms:.2f}ms")


async def test_streaming(image_base64: str):
    """Test streaming audio output."""
    print("\n" + "=" * 60)
    print("TEST: Streaming Audio with Timing")
    print("=" * 60)

    narrator = GeminiLiveNarrator()
    scene = MOCK_SCENES["crowded"]

    print(f"\nScene: {scene['summary']}")
    print("\nStreaming audio chunks:")
    print("-" * 40)

    total_bytes = 0
    chunk_count = 0

    async for audio_chunk, elapsed_ms in narrator.narrate_with_image_stream(scene, image_base64):
        chunk_count += 1
        total_bytes += len(audio_chunk)
        print(f"[{elapsed_ms:>6.0f}ms] Chunk {chunk_count}: {len(audio_chunk)} bytes")

    duration_sec = total_bytes / (24000 * 2)
    print("-" * 40)
    print(f"Total: {chunk_count} chunks, {total_bytes/1024:.1f}KB, {duration_sec:.1f}s audio")


def test_sync_wrapper(image_base64: str):
    """Test synchronous wrapper."""
    print("\n" + "=" * 60)
    print("TEST: Sync GeminiNarrator Wrapper")
    print("=" * 60)

    narrator = GeminiNarrator()
    scene = MOCK_SCENES["emergency"]

    print(f"\nScene: {scene['summary']} (EMERGENCY)")

    result = narrator.narrate_with_image(scene, image_base64)

    audio_kb = len(result.audio_data) / 1024
    duration_sec = len(result.audio_data) / (24000 * 2)

    print(f"\nLatency: {result.latency_ms:.0f}ms | Total: {result.total_ms:.0f}ms")
    print(f"Audio: {audio_kb:.1f}KB ({duration_sec:.1f}s)")

    result.save_wav("test_emergency.wav")
    print("Saved: test_emergency.wav")


async def main():
    print("GeminiLiveNarrator Test Suite (Vertex AI)")
    print("Model: gemini-live-2.5-flash-native-audio")
    print("Output: Audio (24kHz, 16-bit PCM)")
    print()

    # Get test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if Path(image_path).exists():
            print(f"Using image: {image_path}")
            image_base64 = load_image(image_path)
        else:
            print(f"File not found: {image_path}, using generated test image")
            image_base64 = create_test_image()
    else:
        print("No image provided, using generated test image")
        print("Usage: python test_gemini_service.py [path_to_image.jpg]")
        image_base64 = create_test_image()

    print()

    # Run tests
    await test_async_narrator(image_base64)
    await test_streaming(image_base64)
    test_sync_wrapper(image_base64)

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("Play audio files: aplay test_output.wav")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
