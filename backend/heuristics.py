"""
Heuristics Engine for Project Helios v2

Decides WHEN to call Gemini based on YOLO detection data.
Gemini then decides WHAT to say.

Motion-aware: When user is stationary, only speaks for emergencies,
approaching objects, or genuinely new information.
"""

from dataclasses import dataclass, field
from typing import Set
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
    info_debounce_seconds: float = 10.0  # Was 3.0 - increased to reduce chatter

    # Debounce: minimum seconds between any non-urgent speech
    min_speech_interval_seconds: float = 2.0  # Don't speak more than once per 2s

    # Motion: When stationary, be much quieter (only emergencies/approaching)
    stationary_info_debounce_seconds: float = 30.0  # Rarely speak when still
    stationary_skip_path_hazards: bool = True  # Don't warn about static obstacles

    # Objects that warrant proactive speaking (reduced set - only real hazards/goals)
    important_objects: Set[str] = field(default_factory=lambda: {
        # Navigation landmarks
        "door", "stairs", "elevator",
        # Vehicles (safety critical)
        "car", "truck", "bus", "motorcycle",
        # People (social awareness)
        "person",
    })

    # Objects that are hazards when in path (triggers GUIDANCE)
    path_hazards: Set[str] = field(default_factory=lambda: {
        "chair", "table", "bench", "couch", "sofa",
        "bicycle", "dog", "cat",
        "suitcase", "backpack", "box",
    })

    # Positions considered "in walking path"
    path_positions: Set[str] = field(default_factory=lambda: {
        "center", "mid-center", "bottom-center"
    })

    # Distance thresholds (if using numeric distances)
    immediate_threshold_feet: float = 3.0
    close_threshold_feet: float = 8.0


# Default sets for use without config
IMPORTANT_OBJECTS: Set[str] = {
    # Navigation landmarks
    "door", "stairs", "elevator",
    # Vehicles (safety critical)
    "car", "truck", "bus", "motorcycle",
    # People (social awareness)
    "person",
}

PATH_HAZARDS: Set[str] = {
    "chair", "table", "bench", "couch", "sofa",
    "bicycle", "dog", "cat",
    "suitcase", "backpack", "box",
}

PATH_POSITIONS: Set[str] = {"center", "mid-center", "bottom-center"}


def evaluate_scene(
    scene_analysis: dict,
    recent_objects: Set[str],
    last_spoke_seconds_ago: float,
    config: HeuristicsConfig | None = None
) -> SpeakDecision:
    """
    Evaluate YOLO scene data and decide if Gemini should be called.

    Args:
        scene_analysis: YOLO detection results with structure:
            {
                "objects": [{"label": str, "distance": str, "position": str}, ...],
                "emergency_stop": bool,
                "motion": {
                    "is_moving": bool,           # From step detector/accelerometer
                    "speed_mps": float | None,   # Meters per second (optional)
                    "speed_avg_1s_mps": float | None,
                    "steps_last_3s": int,        # Steps in last ~3 seconds (optional)
                    "velocity_x_mps": float | None,
                    "velocity_z_mps": float | None,
                    "magnetic_x_ut": int | None,
                    "magnetic_z_ut": int | None,
                }
            }
        recent_objects: Object labels seen in last N seconds (for debouncing)
        last_spoke_seconds_ago: Time since last speech output
        config: Optional HeuristicsConfig for tuning (uses defaults if None)

    Returns:
        SpeakDecision with should_speak, urgency, and reason
    """
    if config is None:
        config = HeuristicsConfig()

    objects = scene_analysis.get("objects", [])
    emergency = scene_analysis.get("emergency_stop", False)

    # Motion awareness: default to moving (safer - will speak more)
    motion = scene_analysis.get("motion", {})
    is_moving = motion.get("is_moving", True)

    # URGENT: Emergency flag - highest priority (always speaks immediately)
    if emergency:
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.URGENT,
            reason="Emergency detected"
        )

    # ALERT: Immediate distance (< 3 feet) - very high priority
    # Only for actual hazards, not every object
    immediate_objects = [o for o in objects if o.get("distance") == "immediate"]
    if immediate_objects:
        # Filter to only hazardous immediate objects
        hazardous_immediate = [
            o for o in immediate_objects
            if o.get("label") in config.important_objects
            or o.get("label") in config.path_hazards
            or o.get("label") in {"wall", "pole", "pillar", "obstacle"}
        ]
        if hazardous_immediate:
            labels = [o.get("label", "object") for o in hazardous_immediate]
            return SpeakDecision(
                should_speak=True,
                urgency=UrgencyLevel.ALERT,
                reason=f"Immediate: {', '.join(labels)}"
            )

    # Global debounce: don't speak too frequently (except URGENT)
    if last_spoke_seconds_ago < config.min_speech_interval_seconds:
        return SpeakDecision(
            should_speak=False,
            urgency=UrgencyLevel.SILENT,
            reason="Clear path, no changes"
        )

    # GUIDANCE: Close HAZARDS in walking path (not just any object)
    # Skip when stationary - user isn't walking into static obstacles
    if is_moving or not config.stationary_skip_path_hazards:
        close_hazards_in_path = [
            o for o in objects
            if o.get("distance") == "close"
            and any(p in o.get("position", "") for p in config.path_positions)
            and o.get("label") in config.path_hazards
        ]
        if close_hazards_in_path:
            labels = [o.get("label", "object") for o in close_hazards_in_path]
            return SpeakDecision(
                should_speak=True,
                urgency=UrgencyLevel.GUIDANCE,
                reason=f"In path: {', '.join(labels)}"
            )

    # INFO: New important objects (debounced heavily to avoid spam)
    # Use longer debounce when stationary
    info_debounce = (
        config.stationary_info_debounce_seconds
        if not is_moving
        else config.info_debounce_seconds
    )
    if last_spoke_seconds_ago > info_debounce:
        current_labels = {o.get("label") for o in objects if o.get("label")}
        new_important = (current_labels & config.important_objects) - recent_objects

        if new_important:
            return SpeakDecision(
                should_speak=True,
                urgency=UrgencyLevel.INFO,
                reason=f"New: {', '.join(sorted(new_important))}"
            )

    # SILENT: Nothing noteworthy
    return SpeakDecision(
        should_speak=False,
        urgency=UrgencyLevel.SILENT,
        reason="Clear path, no changes"
    )
