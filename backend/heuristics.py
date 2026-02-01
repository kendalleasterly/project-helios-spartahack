"""
Heuristics Engine for Project Helios v2

Decides WHEN to call Gemini based on YOLO detection data.
Gemini then decides WHAT to say.

Motion-aware: When user is stationary, only speaks for emergencies,
approaching objects, or genuinely new information.
"""

from dataclasses import dataclass, field
from typing import Set, Optional
from enum import Enum


class UrgencyLevel(Enum):
    """Urgency levels for speak decisions, ordered from lowest to highest."""
    SILENT = "silent"
    INFO = "info"
    GUIDANCE = "guidance"
    ALERT = "alert"
    URGENT = "urgent"


@dataclass
class SpeakDecision:
    """Result of heuristics evaluation."""
    should_speak: bool
    urgency: UrgencyLevel
    reason: str


@dataclass
class HeuristicsConfig:
    """Tunable parameters for the heuristics engine."""
    # Debounce: minimum seconds between INFO-level speech
    info_debounce_seconds: float = 10.0

    # Debounce: minimum seconds between any non-urgent speech
    min_speech_interval_seconds: float = 2.0

    # Motion: When stationary, be much quieter
    stationary_info_debounce_seconds: float = 30.0
    stationary_skip_path_hazards: bool = True

    # Fast movement threshold (m/s)
    fast_movement_threshold_mps: float = 0.1

    # Objects that warrant proactive speaking
    important_objects: Set[str] = field(default_factory=lambda: {
        "door", "stairs", "elevator",
        "car", "truck", "bus", "motorcycle",
        "person",
    })

    # Objects that are hazards when in path
    path_hazards: Set[str] = field(default_factory=lambda: {
        "chair", "table", "bench", "couch", "sofa",
        "bicycle", "dog", "cat",
        "suitcase", "backpack", "box",
        "pole", "pillar", "wall", "obstacle"
    })

    # Positions considered "in walking path"
    # Now uses the oval intersection logic ("path") instead of grid
    path_positions: Set[str] = field(default_factory=lambda: {"path"})

    # Distance thresholds
    immediate_threshold_feet: float = 3.0
    close_threshold_feet: float = 8.0


# Default sets for use without config
IMPORTANT_OBJECTS: Set[str] = {
    "door", "stairs", "elevator",
    "car", "truck", "bus", "motorcycle",
    "person",
}

PATH_HAZARDS: Set[str] = {
    "chair", "table", "bench", "couch", "sofa",
    "bicycle", "dog", "cat",
    "suitcase", "backpack", "box",
    "pole", "pillar", "wall", "obstacle"
}

PATH_POSITIONS: Set[str] = {"path"}


# Objects that trigger immediate STOP warnings (IDs: 0, 13, 56, 57, 62)
WARNING_TRIGGER_LABELS: Set[str] = {
    "person", "bench", "chair", "couch", "sofa", "tv", "monitor"
}

def evaluate_scene(
    scene_analysis: dict,
    recent_objects: Set[str],
    last_spoke_seconds_ago: float,
    config: Optional[HeuristicsConfig] = None
) -> SpeakDecision:
    """
    Evaluate YOLO scene data and decide if Gemini should be called.
    DEBUG MODE: If object in oval AND moving > 0.1 m/s AND object is dangerous, YELL.
    """
    if config is None:
        config = HeuristicsConfig()

    objects = scene_analysis.get("objects", [])
    motion = scene_analysis.get("motion", {})
    
    # 1. Determine if moving
    speed = motion.get("speed_mps", 0.0) or 0.0
    steps = motion.get("steps_last_3s", 0) or 0
    is_moving = speed > 0.1 or steps > 0
    
    # 2. Check for DANGEROUS objects in the path
    # Must be in the oval ("path") AND be in our specific warning list
    path_objects = [
        o for o in objects
        if o.get("position") == "path" 
        and o.get("label") in WARNING_TRIGGER_LABELS
    ]
    
    # 3. Trigger only if BOTH are true
    if is_moving and path_objects:
        labels = [o.get("label", "object") for o in path_objects]
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.URGENT,
            reason=f"STOP: {', '.join(labels)} ahead"
        )

    # Otherwise silent
    return SpeakDecision(
        should_speak=False,
        urgency=UrgencyLevel.SILENT,
        reason="Clear path or stationary"
    )
