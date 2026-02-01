"""
Heuristics Engine for Project Helios v2

Decides WHEN to call Gemini based on YOLO detection data.
Gemini then decides WHAT to say.
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
    info_debounce_seconds: float = 3.0

    # Objects that warrant proactive speaking
    important_objects: Set[str] = field(default_factory=lambda: {
        # Navigation
        "door", "stairs", "elevator", "escalator",
        # Seating
        "chair", "couch", "bench", "sofa",
        # Safety
        "car", "truck", "bus", "motorcycle", "bicycle",
        # People
        "person",
        # Hazards
        "dog", "cat",
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
    # Navigation
    "door", "stairs", "elevator", "escalator",
    # Seating
    "chair", "couch", "bench", "sofa",
    # Safety
    "car", "truck", "bus", "motorcycle", "bicycle",
    # People
    "person",
    # Hazards
    "dog", "cat",
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
                "emergency_stop": bool
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

    # URGENT: Emergency flag - highest priority
    if emergency:
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.URGENT,
            reason="Emergency detected"
        )

    # ALERT: Immediate distance (< 3 feet) - very high priority
    immediate_objects = [o for o in objects if o.get("distance") == "immediate"]
    if immediate_objects:
        labels = [o.get("label", "object") for o in immediate_objects]
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.ALERT,
            reason=f"Immediate: {', '.join(labels)}"
        )

    # GUIDANCE: Close objects in walking path
    close_in_path = [
        o for o in objects
        if o.get("distance") == "close"
        and any(p in o.get("position", "") for p in config.path_positions)
    ]
    if close_in_path:
        labels = [o.get("label", "object") for o in close_in_path]
        return SpeakDecision(
            should_speak=True,
            urgency=UrgencyLevel.GUIDANCE,
            reason=f"In path: {', '.join(labels)}"
        )

    # INFO: New important objects (debounced to avoid spam)
    if last_spoke_seconds_ago > config.info_debounce_seconds:
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
