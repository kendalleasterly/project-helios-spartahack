"""
Test script for GeminiNarrator with mock scene_analysis data.

Run: python test_gemini_service.py
"""

import time
from gemini_service import GeminiNarrator

# Mock scene_analysis data (matches YOLO server output format)
MOCK_SCENES = {
    "empty": {
        "timestamp": "2026-01-31T12:00:00.000Z",
        "emergency_stop": False,
        "summary": "",
        "objects": []
    },
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


def test_blocking():
    """Test blocking narrate() with cache verification."""
    print("=" * 50)
    print("TEST: Blocking narrate() with caching")
    print("=" * 50)

    narrator = GeminiNarrator()

    for name, scene in MOCK_SCENES.items():
        print(f"\n--- Scene: {name} ---")

        # First call (cache miss)
        start = time.perf_counter()
        result = narrator.narrate(scene)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[MISS] {elapsed:.0f}ms: {result}")

        # Second call (cache hit)
        start = time.perf_counter()
        result = narrator.narrate(scene)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[HIT]  {elapsed:.2f}ms: {result}")


def test_streaming():
    """Test streaming narrate_stream() output."""
    print("\n" + "=" * 50)
    print("TEST: Streaming narrate_stream()")
    print("=" * 50)

    narrator = GeminiNarrator()
    scene = MOCK_SCENES["crowded"]

    print("\n--- Streaming output ---")
    start = time.perf_counter()
    first_chunk_time = None

    for chunk in narrator.narrate_stream(scene):
        if first_chunk_time is None:
            first_chunk_time = (time.perf_counter() - start) * 1000
        print(chunk, end="", flush=True)

    total_time = (time.perf_counter() - start) * 1000
    print(f"\n\nFirst chunk: {first_chunk_time:.0f}ms | Total: {total_time:.0f}ms")


def test_cache_stats():
    """Show cache behavior across repeated calls."""
    print("\n" + "=" * 50)
    print("TEST: Cache statistics")
    print("=" * 50)

    narrator = GeminiNarrator()
    scene = MOCK_SCENES["person_ahead"]

    times = []
    for i in range(5):
        start = time.perf_counter()
        narrator.narrate(scene)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        status = "MISS" if i == 0 else "HIT"
        print(f"Call {i+1}: [{status}] {elapsed:.2f}ms")

    print(f"\nFirst call: {times[0]:.0f}ms")
    print(f"Cached avg: {sum(times[1:]) / len(times[1:]):.2f}ms")


if __name__ == "__main__":
    print("GeminiNarrator Test Suite")
    print("Using model: gemini-2.0-flash\n")

    test_blocking()
    test_streaming()
    test_cache_stats()

    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)
