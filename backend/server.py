"""
YOLO11 Vision Processing Server for Blind Assistant
Receives video frames via Socket.IO, processes with YOLO11, returns semantic spatial data
"""

import asyncio
import base64
import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import networkx as nx
import numpy as np
import socketio
import sounddevice as sd
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from PIL import Image
from ultralytics import YOLO

from gemini_service import GeminiLiveNarrator, NarrationResult
from contextual_gemini_service import GeminiContextualNarrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="YOLO11 Vision Server")

# Initialize Socket.IO with CORS enabled
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Allow all origins for development
    logger=False,
    engineio_logger=False,
    max_http_buffer_size=1e7
)

# Wrap with ASGI app
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app
)

# Global YOLO model (loaded at startup)
yolo_model = None

# Global Gemini narrators (loaded at startup)
gemini_narrator = None  # Legacy Live API narrator
contextual_narrator = None  # New streaming text narrator

# Vehicle classes that trigger emergency warnings
VEHICLE_CLASSES = {'car', 'bus', 'truck', 'motorcycle', 'bicycle'}

# Emergency distance thresholds
EMERGENCY_DISTANCES = {'immediate', 'close'}

# Debug Recording Configuration
SAVE_DEBUG_FRAMES = os.getenv('SAVE_DEBUG_FRAMES', 'true').lower() == 'true'
DEBUG_OUTPUT_DIR = Path('server_debug_output')


def load_yolo_model():
    """
    Load YOLO11 Extra Large model at startup with GPU acceleration.

    YOLO11x (Extra Large) Configuration:
    - Maximum feature extraction capability
    - Highest recall for small/distant objects (lecture hall chairs)
    - Optimized for RTX 4070 with 8GB VRAM
    - Prioritizes detecting EVERYTHING over precision
    """
    global yolo_model
    logger.info("Loading YOLO11 Extra Large model (Maximum Recall Mode)...")

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available! Falling back to CPU (will be slow)")
            device = 'cpu'
        else:
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU Detected: {gpu_name}")

        # Load YOLO11 Extra Large model (maximum recall)
        # ultralytics will auto-download yolo11x.pt (~140MB) if not present
        yolo_model = YOLO('yolo11x.pt')

        # Move model to GPU
        yolo_model.to(device)

        logger.info(f"✓ YOLO11x (Extra Large) loaded successfully on {device.upper()}")

        if device == 'cuda':
            # Log VRAM usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"✓ GPU Memory Allocated: {allocated:.2f} GB")
            logger.info(f"✓ Max Recall Mode: conf=0.05, iou=0.6")

    except Exception as e:
        logger.error(f"✗ Failed to load YOLO11 model: {e}")
        raise


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a Base64 encoded JPEG image to OpenCV format (BGR).

    Args:
        base64_string: Base64 encoded image string

    Returns:
        numpy.ndarray: Image in OpenCV BGR format
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image then to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return opencv_image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise


def calculate_3x3_position(bbox_center_x: float, bbox_center_y: float,
                           image_width: int, image_height: int) -> str:
    """
    Determine position in a 3x3 grid system for high-fidelity spatial awareness.

    3x3 Grid Spatial Logic:

    X-Axis (Horizontal):
    - Left:   0% - 33% of image width
    - Center: 33% - 66% of image width
    - Right:  66% - 100% of image width

    Y-Axis (Vertical):
    - Top:    0% - 33% of image height (sky, ceiling, signs)
    - Middle: 33% - 66% of image height (eye-level, torso)
    - Bottom: 66% - 100% of image height (floor, curbs, ground objects)

    Combined Output Examples:
    - "top-left", "top-center", "top-right"
    - "mid-left", "mid-center", "mid-right"
    - "bottom-left", "bottom-center", "bottom-right"

    Why This Matters:
    A "Chandelier" at top-center vs "Carpet" at bottom-center are critically
    different for navigation. Vertical awareness prevents collisions with
    overhead obstacles and low-lying trip hazards.

    Args:
        bbox_center_x: X-coordinate of bounding box center
        bbox_center_y: Y-coordinate of bounding box center
        image_width: Total width of the image
        image_height: Total height of the image

    Returns:
        str: Grid position like "top-left", "mid-center", "bottom-right"
    """
    # Calculate relative positions (0.0 to 1.0)
    relative_x = bbox_center_x / image_width
    relative_y = bbox_center_y / image_height

    # Determine horizontal zone
    if relative_x < 0.33:
        horizontal = "left"
    elif relative_x < 0.66:
        horizontal = "center"
    else:
        horizontal = "right"

    # Determine vertical zone
    if relative_y < 0.33:
        vertical = "top"
    elif relative_y < 0.66:
        vertical = "mid"
    else:
        vertical = "bottom"

    # Combine into grid position
    return f"{vertical}-{horizontal}"


def calculate_distance(bbox_height: float, image_height: int) -> str:
    """
    Estimate object distance based on bounding box height.

    Distance Logic (Depth Estimation):
    - Immediate: Box height > 50% of image height (RIGHT IN FRONT)
    - Close:     Box height > 20% of image height
    - Far:       Box height ≤ 20% of image height

    Rationale: Larger objects in the frame are closer to the camera.

    Args:
        bbox_height: Height of the bounding box
        image_height: Total height of the image

    Returns:
        str: "immediate", "close", or "far"
    """
    relative_height = bbox_height / image_height

    if relative_height > 0.50:
        return "immediate"
    elif relative_height > 0.20:
        return "close"
    else:
        return "far"


def annotate_frame(image: np.ndarray, objects: List[Dict[str, Any]],
                   emergency_stop: bool = False) -> np.ndarray:
    """
    Draw bounding boxes, labels, and 3x3 grid positions on the image for debugging.

    Visual Coding:
    - Green boxes: Safe objects (far distance)
    - Yellow boxes: Close objects
    - Red boxes: Immediate danger or emergency vehicles

    Each box shows:
    - Label + Confidence (e.g., "Person 95%")
    - 3x3 Grid Position (e.g., "Mid-Center")
    - Distance indicator

    Args:
        image: Original OpenCV image (BGR format)
        objects: List of detected objects with positions and boxes
        emergency_stop: Whether emergency was triggered

    Returns:
        np.ndarray: Annotated image
    """
    # Create a copy to avoid modifying original
    annotated = image.copy()

    # Add emergency banner if triggered
    if emergency_stop:
        # Draw red banner at top
        cv2.rectangle(annotated, (0, 0), (image.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(
            annotated,
            "EMERGENCY: VEHICLE DETECTED",
            (10, 35),
            cv2.FONT_HERSHEY_BOLD,
            1.2,
            (255, 255, 255),
            3
        )

    for obj in objects:
        # Extract object data
        label = obj.get('label', 'unknown')
        confidence = obj.get('confidence', 0.0)
        position = obj.get('position', 'unknown')
        distance = obj.get('distance', 'unknown')
        box = obj.get('box', [0, 0, 0, 0])

        x1, y1, x2, y2 = box

        # Color code by distance and emergency status
        if distance == 'immediate':
            color = (0, 0, 255)  # Red - BGR format
            thickness = 3
        elif distance == 'close':
            color = (0, 165, 255)  # Orange
            thickness = 2
        else:
            color = (0, 255, 0)  # Green
            thickness = 2

        # Override to red if it's a vehicle causing emergency
        if label in VEHICLE_CLASSES and distance in EMERGENCY_DISTANCES:
            color = (0, 0, 255)  # Red
            thickness = 4

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_text = f"{label.capitalize()} {int(confidence * 100)}%"
        position_text = position.replace('-', ' ').title()
        distance_text = distance.upper()

        # Calculate text position (above bounding box)
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

        # Draw label background for readability
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            annotated,
            (x1, text_y - text_height - 5),
            (x1 + text_width + 10, text_y + baseline),
            color,
            -1  # Filled rectangle
        )

        # Draw label text (white text on colored background)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            2
        )

        # Draw position text below the box
        position_y = y2 + 20
        cv2.rectangle(
            annotated,
            (x1, position_y - 15),
            (x1 + 150, position_y + 5),
            (0, 0, 0),  # Black background
            -1
        )
        cv2.putText(
            annotated,
            f"{position_text} | {distance_text}",
            (x1 + 5, position_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White
            1
        )

    # Add object count in top-right corner
    count_text = f"Objects: {len(objects)}"
    cv2.putText(
        annotated,
        count_text,
        (image.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return annotated


def calculate_centroid(obj: Dict[str, Any]) -> tuple:
    """
    Calculate the centroid (center point) of a bounding box.

    Args:
        obj: Object dict with 'box' field [x1, y1, x2, y2]

    Returns:
        tuple: (center_x, center_y)
    """
    x1, y1, x2, y2 = obj['box']
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculate Euclidean distance between two points.

    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

    Args:
        point1: (x1, y1)
        point2: (x2, y2)

    Returns:
        float: Distance in pixels
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def detect_crowd_clusters(objects: List[Dict[str, Any]],
                         proximity_threshold: float = 200.0,
                         image_width: int = 1280,
                         image_height: int = 720) -> List[Dict[str, Any]]:
    """
    Graph-based "Mega-Clustering" using NetworkX Connected Components.

    Algorithm (The "Mega-Cluster" Engine):
    1. Build a graph G where each object is a node
    2. Add edges between objects within proximity_threshold pixels
    3. Find connected components (chain reactions: A→B→C merge into one cluster)
    4. Analyze bounding box union of each component
    5. Generate dynamic descriptions based on spatial coverage

    Mega-Cluster Logic:
    - If cluster spans > 40% of image width → "Massive arrangement spanning [X] to [Y]"
    - If cluster contains > 10 items → "Large group of [N] [Label]s"
    - Refined spatial naming based on bounding box union coverage

    Args:
        objects: List of detected objects with bounding boxes
        proximity_threshold: Max distance (pixels) for edge creation (default 200px for lecture halls)
        image_width: Width of the image (for spatial analysis)
        image_height: Height of the image (for spatial analysis)

    Returns:
        List of mega-cluster metadata dicts with dynamic descriptions
    """
    from collections import defaultdict

    # Group objects by label (process each class separately)
    groups = defaultdict(list)
    for obj in objects:
        groups[obj['label']].append(obj)

    clusters = []

    for label, group in groups.items():
        if len(group) < 3:
            continue  # Need at least 3 for a cluster

        # Step 1: Build a graph with objects as nodes
        G = nx.Graph()

        # Add all objects as nodes
        for idx, obj in enumerate(group):
            G.add_node(idx, obj=obj)

        # Step 2: Add edges between nearby objects (proximity-based connections)
        for i, obj_i in enumerate(group):
            centroid_i = calculate_centroid(obj_i)

            for j, obj_j in enumerate(group):
                if i < j:  # Avoid duplicate edges
                    centroid_j = calculate_centroid(obj_j)
                    dist = euclidean_distance(centroid_i, centroid_j)

                    if dist < proximity_threshold:
                        G.add_edge(i, j)

        # Step 3: Find connected components (mega-clusters via chain reactions)
        components = list(nx.connected_components(G))

        for component in components:
            if len(component) < 3:
                continue  # Still need at least 3 for a cluster

            # Get all objects in this mega-cluster
            cluster_objects = [group[idx] for idx in component]

            # Step 4: Calculate bounding box union (min_x, min_y, max_x, max_y)
            all_boxes = [obj['box'] for obj in cluster_objects]
            min_x = min(box[0] for box in all_boxes)
            min_y = min(box[1] for box in all_boxes)
            max_x = max(box[2] for box in all_boxes)
            max_y = max(box[3] for box in all_boxes)

            # Calculate union box dimensions and coverage
            union_width = max_x - min_x
            union_height = max_y - min_y
            width_coverage = union_width / image_width  # Fraction of image width
            height_coverage = union_height / image_height

            # Step 5: Determine spatial zones covered by the union box
            # Calculate center of union box for primary position
            union_center_x = (min_x + max_x) / 2
            union_center_y = (min_y + max_y) / 2

            # Determine which horizontal zones are covered
            left_edge = min_x / image_width
            right_edge = max_x / image_width

            zones_covered = []
            if left_edge < 0.33:
                zones_covered.append("left")
            if left_edge < 0.66 and right_edge > 0.33:
                zones_covered.append("center")
            if right_edge > 0.66:
                zones_covered.append("right")

            # Determine vertical coverage
            top_edge = min_y / image_height
            bottom_edge = max_y / image_height

            vertical_zones = []
            if top_edge < 0.33:
                vertical_zones.append("top")
            if top_edge < 0.66 and bottom_edge > 0.33:
                vertical_zones.append("mid")
            if bottom_edge > 0.66:
                vertical_zones.append("bottom")

            # Build spatial description
            if len(zones_covered) >= 2:
                # Spans multiple horizontal zones
                span_description = f"from {zones_covered[0]} to {zones_covered[-1]}"
            else:
                span_description = zones_covered[0] if zones_covered else "center"

            # Add vertical context if it spans multiple zones
            if len(vertical_zones) >= 2:
                vertical_description = f"{vertical_zones[0]} to {vertical_zones[-1]}"
            else:
                vertical_description = vertical_zones[0] if vertical_zones else "mid"

            # Step 6: Generate dynamic cluster description
            count = len(component)

            # Determine cluster type based on size and coverage
            if width_coverage > 0.4:
                # Massive arrangement spanning significant portion of frame
                cluster_type = 'massive_arrangement'
                description = f"spanning {span_description}"
            elif count > 10:
                # Large group
                cluster_type = 'large_group'
                description = f"taking up the {span_description}"
            else:
                # Dense cluster (default)
                cluster_type = 'crowd'
                # Combine vertical and horizontal for precise location
                if len(zones_covered) == 1 and len(vertical_zones) == 1:
                    description = f"{vertical_description} {span_description}"
                else:
                    description = f"covering {vertical_description} {span_description}"

            clusters.append({
                'type': cluster_type,
                'label': label,
                'count': count,
                'description': description,
                'width_coverage': width_coverage,
                'height_coverage': height_coverage,
                'bounding_box': [min_x, min_y, max_x, max_y],
                'centroid': (union_center_x, union_center_y)
            })

    return clusters


def detect_row_patterns(objects: List[Dict[str, Any]],
                       y_tolerance: float = 50.0,
                       min_count: int = 3) -> List[Dict[str, Any]]:
    """
    Detect horizontal "row" patterns: 3+ objects aligned horizontally.

    Row Detection Algorithm:
    1. Group objects by class label
    2. Calculate Y-coordinate variance (vertical spread)
    3. If variance is LOW (similar Y) but X-spread is HIGH → it's a row
    4. Extract leftmost and rightmost positions for description

    Geometric Math:
    - y_tolerance = 50px means centroids must be within 50px vertically
    - This accounts for slight misalignments in real-world scenes
    - X-spread measures horizontal span (left-to-right extent)

    Example: "A row of 4 chairs spanning from left to center"

    Args:
        objects: List of detected objects
        y_tolerance: Max Y-coordinate variance for row detection (pixels)
        min_count: Minimum objects to form a row

    Returns:
        List of row pattern metadata dicts
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for obj in objects:
        groups[obj['label']].append(obj)

    rows = []

    for label, group in groups.items():
        if len(group) < min_count:
            continue

        # Calculate centroids and Y-coordinates
        centroids = [calculate_centroid(obj) for obj in group]
        y_coords = [c[1] for c in centroids]
        x_coords = [c[0] for c in centroids]

        # Check if Y-variance is low (horizontally aligned)
        y_variance = np.var(y_coords)
        x_spread = max(x_coords) - min(x_coords)

        # Row criteria: low Y-variance, high X-spread
        if y_variance < (y_tolerance ** 2) and x_spread > 200:
            # Determine span (leftmost to rightmost position)
            positions = [obj['position'] for obj in group]
            leftmost = min(x_coords)
            rightmost = max(x_coords)

            # Extract horizontal zones
            left_zone = "left" if any("left" in p for p in positions) else None
            center_zone = "center" in " ".join(positions)
            right_zone = "right" if any("right" in p for p in positions) else None

            # Build span description
            span_parts = []
            if left_zone:
                span_parts.append("left")
            if center_zone:
                span_parts.append("center")
            if right_zone:
                span_parts.append("right")

            span = " to ".join(span_parts) if len(span_parts) > 1 else span_parts[0]

            rows.append({
                'type': 'row',
                'label': label,
                'count': len(group),
                'span': span,
                'y_variance': y_variance
            })

    return rows


def detect_stack_patterns(objects: List[Dict[str, Any]],
                         x_tolerance: float = 50.0,
                         min_count: int = 2) -> List[Dict[str, Any]]:
    """
    Detect vertical "stack" patterns: 2+ objects aligned vertically.

    Stack Detection Algorithm:
    1. Group objects by class label
    2. Calculate X-coordinate variance (horizontal spread)
    3. If variance is LOW (similar X) but Y-spread is HIGH → it's a stack
    4. Report vertical position (typically "mid" or "bottom")

    Geometric Math:
    - x_tolerance = 50px means centroids must be within 50px horizontally
    - Y-spread measures vertical extent (top-to-bottom)

    Example: "A stack of 3 boxes (right side)"

    Args:
        objects: List of detected objects
        x_tolerance: Max X-coordinate variance for stack detection (pixels)
        min_count: Minimum objects to form a stack

    Returns:
        List of stack pattern metadata dicts
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for obj in objects:
        groups[obj['label']].append(obj)

    stacks = []

    for label, group in groups.items():
        if len(group) < min_count:
            continue

        # Calculate centroids and coordinates
        centroids = [calculate_centroid(obj) for obj in group]
        x_coords = [c[0] for c in centroids]
        y_coords = [c[1] for c in centroids]

        # Check if X-variance is low (vertically aligned)
        x_variance = np.var(x_coords)
        y_spread = max(y_coords) - min(y_coords)

        # Stack criteria: low X-variance, high Y-spread
        if x_variance < (x_tolerance ** 2) and y_spread > 100:
            # Determine horizontal position (use most common)
            positions = [obj['position'] for obj in group]
            horizontal = max(set(p.split('-')[1] for p in positions),
                           key=lambda x: sum(1 for p in positions if x in p))

            stacks.append({
                'type': 'stack',
                'label': label,
                'count': len(group),
                'position': horizontal,
                'x_variance': x_variance
            })

    return stacks


def process_detections(results, image_height: int, image_width: int) -> Dict[str, Any]:
    """
    Convert YOLO detections to high-fidelity semantic spatial data with clustering.

    Enhanced Pipeline:
    1. Extract raw detections with 3x3 grid positions
    2. Detect geometric patterns (crowds, rows, stacks)
    3. Assign cluster IDs to objects
    4. Flag emergency situations

    Args:
        results: YOLO inference results
        image_height: Height of input image
        image_width: Width of input image

    Returns:
        dict: Enhanced JSON with objects, clusters, and emergency flag
    """
    objects = []
    emergency_stop = False

    # Step 1: Extract detections from YOLO results with 3x3 grid positions
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes

        for idx, box in enumerate(boxes):
            # Extract bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate bounding box center and dimensions
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Get class label and confidence
            class_id = int(box.cls[0])
            label = results[0].names[class_id].lower()
            confidence = float(box.conf[0])

            # Calculate 3x3 grid position (NEW: includes vertical awareness)
            position = calculate_3x3_position(
                bbox_center_x, bbox_center_y,
                image_width, image_height
            )

            # Calculate distance (depth estimation)
            distance = calculate_distance(bbox_height, image_height)

            # Create enhanced object entry
            obj = {
                "id": idx,  # Unique ID for cluster tracking
                "label": label,
                "confidence": round(confidence, 2),
                "position": position,  # Now 3x3 grid (e.g., "top-left", "mid-center")
                "distance": distance,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "cluster_id": None  # Will be assigned if part of a cluster
            }
            objects.append(obj)

            # Emergency Detection Logic
            # Trigger if vehicle is dangerously close
            if label in VEHICLE_CLASSES and distance in EMERGENCY_DISTANCES:
                emergency_stop = True
                logger.warning(
                    f"⚠️  EMERGENCY: {label} detected at {distance} distance ({position})"
                )

    # Step 2: Detect geometric patterns using NetworkX mega-clustering
    crowd_clusters = detect_crowd_clusters(
        objects,
        proximity_threshold=200.0,  # Generous threshold for lecture hall rows
        image_width=image_width,
        image_height=image_height
    )
    row_patterns = detect_row_patterns(objects)
    stack_patterns = detect_stack_patterns(objects)

    # Combine all detected patterns
    all_clusters = crowd_clusters + row_patterns + stack_patterns

    # Step 3: Assign cluster IDs to objects (optional, for debugging)
    # For simplicity, we'll use the cluster metadata in summary generation
    # rather than modifying the objects array

    return {
        "objects": objects,
        "clusters": all_clusters,  # NEW: Geometric pattern metadata
        "emergency_stop": emergency_stop
    }


def play_audio_on_server(audio_data: bytes, sample_rate: int = 24000):
    """
    Play PCM audio data through server speakers for testing.

    Args:
        audio_data: Raw PCM audio bytes (16-bit, mono)
        sample_rate: Sample rate in Hz (default: 24000 for Gemini)
    """
    try:
        # Convert bytes to numpy array (int16 PCM format)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float32 in range [-1.0, 1.0] for sounddevice
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Play audio (blocking)
        logger.info(f"🔊 Playing audio on server speakers ({len(audio_data)} bytes, {sample_rate}Hz)...")
        sd.play(audio_float, samplerate=sample_rate, blocking=True)
        logger.info("✓ Audio playback complete")

    except Exception as e:
        logger.error(f"✗ Failed to play audio on server: {e}")


def generate_summary(objects: List[Dict[str, Any]],
                    clusters: List[Dict[str, Any]]) -> str:
    """
    Generate sophisticated natural language scene description for LLM.

    Enhanced Mega-Cluster Strategy:
    1. Prioritize mega-clusters (massive arrangements, large groups) over individual objects
    2. Use dynamic descriptions based on spatial coverage and count
    3. Avoid fragmented descriptions - merge connected objects into cohesive narratives
    4. Keep it concise but informative

    Examples:
    - OLD: "A dense cluster of 3 chairs (mid left), a dense cluster of 3 chairs (mid left)..."
    - NEW: "A massive arrangement of 65 chairs spanning from left to right across the lecture hall."

    Args:
        objects: List of detected objects with 3x3 grid positions
        clusters: List of mega-cluster metadata from NetworkX analysis

    Returns:
        str: Natural language scene description
    """
    if not objects:
        return "Clear path ahead, no objects detected."

    # Track which objects are part of clusters (to avoid double-counting)
    clustered_labels = set()
    summary_parts = []

    # Step 1: Describe mega-clusters first (highest priority)
    for cluster in clusters:
        label = cluster.get('label', 'object')
        count = cluster.get('count', 0)
        cluster_type = cluster.get('type', 'unknown')
        description = cluster.get('description', 'in the frame')

        # Pluralize label
        plural_label = label + 's' if not label.endswith('s') else label

        if cluster_type == 'massive_arrangement':
            # Massive arrangements spanning significant portions of frame
            summary_parts.append(
                f"a massive arrangement of {count} {plural_label} {description}"
            )

        elif cluster_type == 'large_group':
            # Large groups (10+ items)
            summary_parts.append(
                f"a large group of {count} {plural_label} {description}"
            )

        elif cluster_type == 'crowd':
            # Dense clusters (3-10 items)
            summary_parts.append(
                f"a cluster of {count} {plural_label} ({description})"
            )

        elif cluster_type == 'row':
            # Row patterns (legacy support for row detection if still used)
            span = cluster.get('span', 'across the frame')
            summary_parts.append(
                f"a row of {count} {plural_label} spanning from {span}"
            )

        elif cluster_type == 'stack':
            # Stack patterns (legacy support for stack detection if still used)
            pos = cluster.get('position', 'center')
            summary_parts.append(
                f"a stack of {count} {plural_label} ({pos} side)"
            )

        # Mark this label as clustered
        clustered_labels.add(label)

    # Step 2: Describe individual objects (not in clusters)
    individual_objects = [
        obj for obj in objects
        if obj['label'] not in clustered_labels
    ]

    # Group individual objects by label for cleaner descriptions
    from collections import defaultdict
    singles = defaultdict(list)
    for obj in individual_objects:
        singles[obj['label']].append(obj)

    for label, instances in singles.items():
        if len(instances) == 1:
            obj = instances[0]
            position = obj['position'].replace('-', ' ')
            distance = obj['distance']

            # Add distance qualifier for close objects
            distance_phrase = ""
            if distance == "immediate":
                distance_phrase = "directly in front, "
            elif distance == "close":
                distance_phrase = "nearby, "

            summary_parts.append(f"a {label} ({distance_phrase}{position})")

        else:
            # Multiple instances but not clustered (spread out)
            count = len(instances)
            positions = [obj['position'].replace('-', ' ') for obj in instances]
            unique_positions = list(set(positions))

            if len(unique_positions) == 1:
                summary_parts.append(
                    f"{count} {label}s ({unique_positions[0]})"
                )
            else:
                # Scattered across multiple zones
                summary_parts.append(
                    f"{count} {label}s scattered across the frame"
                )

    # Step 3: Combine into natural sentences
    if len(summary_parts) == 0:
        return "Scene contains objects but spatial analysis is incomplete."

    elif len(summary_parts) == 1:
        return summary_parts[0].capitalize() + "."

    elif len(summary_parts) == 2:
        return f"{summary_parts[0].capitalize()}, with {summary_parts[1]}."

    else:
        # 3+ elements: use commas and "and"
        main_part = ", ".join(summary_parts[:-1])
        last_part = summary_parts[-1]
        return f"{main_part.capitalize()}, and {last_part}."


async def call_gemini(scene_analysis: Dict[str, Any], image_base64: str) -> Optional[NarrationResult]:
    """
    Call Gemini Live API for audio narration using Ben's implementation.

    Args:
        scene_analysis: Dict with 'summary', 'objects', 'emergency_stop'
        image_base64: Base64-encoded JPEG image

    Returns:
        NarrationResult with audio data and metrics, or None if Gemini not available
    """
    if gemini_narrator is None:
        logger.warning("⚠️  Gemini narrator not initialized, skipping narration")
        return None

    try:
        logger.info("🤖 Calling Gemini Live API for narration...")
        logger.info(f"   Summary: {scene_analysis.get('summary', 'N/A')}")
        logger.info(f"   Emergency: {scene_analysis.get('emergency_stop', False)}")

        # Call Ben's Gemini implementation
        result = await gemini_narrator.narrate_with_image(
            scene_analysis=scene_analysis,
            image_base64=image_base64
        )

        logger.info(f"✓ Gemini narration complete:")
        logger.info(f"   Latency: {result.latency_ms:.1f}ms")
        logger.info(f"   Total: {result.total_ms:.1f}ms")
        logger.info(f"   Audio size: {len(result.audio_data)} bytes")
        logger.info(f"   Cached: {result.cached}")
        if result.transcript:
            logger.info(f"   Transcript: {result.transcript}")

        return result

    except Exception as e:
        logger.error(f"✗ Gemini narration failed: {e}", exc_info=True)
        return None


@app.on_event("startup")
async def startup_event():
    """Load YOLO model and setup debug output directory when server starts."""
    global gemini_narrator, contextual_narrator

    load_yolo_model()

    # Initialize Contextual Gemini narrator (primary)
    try:
        logger.info("🤖 Initializing Contextual Gemini Narrator...")
        contextual_narrator = GeminiContextualNarrator()
        logger.info("✓ Contextual Narrator initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize contextual narrator: {e}")
        logger.warning("⚠️  Server will continue without Gemini narration")

    # Initialize Legacy Gemini Live narrator (fallback)
    try:
        logger.info("🤖 Initializing Gemini Live Narrator (fallback)...")
        gemini_narrator = GeminiLiveNarrator()
        logger.info("✓ Gemini Live Narrator initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Gemini narrator: {e}")

    # Setup debug output directory
    if SAVE_DEBUG_FRAMES:
        DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
        logger.info(f"💾 Debug frame recording ENABLED → {DEBUG_OUTPUT_DIR.absolute()}")
    else:
        logger.info("💾 Debug frame recording DISABLED")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "YOLO11 Vision Server",
        "model": "yolo11n.pt"
    }


@app.get("/health")
async def health():
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": yolo_model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"✓ Client connected: {sid}")
    await sio.emit('connection_established', {'status': 'connected'}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"✗ Client disconnected: {sid}")


@sio.event
async def video_frame_streaming(sid, data):
    """
    Process video frame with streaming text response (NEW APPROACH).

    This uses the contextual Gemini narrator that:
    1. Decides whether to speak based on context
    2. Streams text tokens to phone for on-device TTS
    3. Maintains conversation history

    Data format:
    {
        'frame': base64_image,
        'user_question': Optional[str],  # User's question if any
        'debug': bool
    }
    """
    try:
        start_time = asyncio.get_event_loop().time()

        # Extract frame
        if 'frame' not in data:
            logger.error("No 'frame' field in received data")
            await sio.emit('error', {'message': 'Missing frame data'}, room=sid)
            return

        base64_frame = data['frame']
        user_question = data.get('user_question', None)
        debug_mode = data.get('debug', False)

        # Step 1: Decode image
        image = decode_base64_image(base64_frame)
        image_height, image_width = image.shape[:2]
        logger.info(f"📷 Frame: {image_width}x{image_height} | Question: {user_question or 'None'}")

        # Step 2: Run YOLO inference
        inference_start = asyncio.get_event_loop().time()

        NAV_CLASSES = [0, 13, 24, 39, 41, 42, 43, 44, 45, 56, 57, 62, 63, 64, 65, 66, 67, 73]

        with torch.no_grad():
            results = yolo_model(
                image,
                verbose=False,
                conf=0.25,
                iou=0.45,
                imgsz=1280,
                classes=NAV_CLASSES,
                agnostic_nms=True
            )

        inference_ms = (asyncio.get_event_loop().time() - inference_start) * 1000
        logger.info(f"⚡ YOLO inference: {inference_ms:.2f}ms")

        # Step 3: Process detections
        detection_data = process_detections(results, image_height, image_width)
        objects = detection_data['objects']
        clusters = detection_data['clusters']
        emergency_stop = detection_data['emergency_stop']

        # Step 4: Generate summary
        summary = generate_summary(objects, clusters)

        # Step 5: Build scene analysis
        scene_analysis = {
            "summary": summary,
            "objects": objects,
            "emergency_stop": emergency_stop
        }

        # Step 6: Stream Gemini response
        if contextual_narrator:
            logger.info("🤖 Calling Contextual Gemini (streaming mode)...")

            token_count = 0
            first_token_time = None

            async for should_speak, text_chunk in contextual_narrator.process_streaming(
                scene_analysis=scene_analysis,
                image_base64=base64_frame,
                user_question=user_question
            ):
                if first_token_time is None:
                    first_token_time = asyncio.get_event_loop().time()
                    gemini_latency = (first_token_time - start_time) * 1000
                    logger.info(f"✓ First token: {gemini_latency:.1f}ms")

                if should_speak:
                    # Stream text token to phone for TTS
                    await sio.emit('text_token', {
                        'token': text_chunk,
                        'emergency': emergency_stop,
                        'is_first': (token_count == 0)
                    }, room=sid)
                    token_count += 1
                else:
                    # Silent response - log but don't send
                    logger.info(f"🔇 SILENT response: {text_chunk}")
                    break

            if token_count > 0:
                logger.info(f"🔊 Streamed {token_count} text tokens to client")

        else:
            logger.warning("⚠️  Contextual narrator not initialized")

        # Step 7: Send debug frame if requested
        if debug_mode:
            annotated_image = annotate_frame(image, objects, emergency_stop)
            _, buffer = cv2.imencode('.jpg', annotated_image)
            debug_base64 = base64.b64encode(buffer).decode('utf-8')

            await sio.emit('debug_frame', {
                'frame': f"data:image/jpeg;base64,{debug_base64}",
                'summary': summary,
                'object_count': len(objects)
            }, room=sid)

        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(f"✓ Total pipeline: {total_time:.1f}ms | Objects: {len(objects)} | Emergency: {emergency_stop}")

    except Exception as e:
        logger.error(f"Error processing video frame: {e}", exc_info=True)
        await sio.emit('error', {
            'message': 'Frame processing failed',
            'error': str(e)
        }, room=sid)


@sio.event
async def video_frame(sid, data):
    """
    Process incoming video frame from iPhone client.

    Pipeline:
    1. Decode Base64 image
    2. Run YOLO11 inference
    3. Convert detections to semantic spatial data
    4. Emit JSON result back to client

    Args:
        sid: Socket session ID
        data: Dictionary containing 'frame' (Base64 encoded image)
    """
    try:
        start_time = asyncio.get_event_loop().time()

        # Extract base64 frame
        if 'frame' not in data:
            logger.error("No 'frame' field in received data")
            await sio.emit('error', {'message': 'Missing frame data'}, room=sid)
            return

        base64_frame = data['frame']
        debug_mode = data.get('debug', False)  # Check for debug flag

        # Step 1: Decode Base64 image
        image = decode_base64_image(base64_frame)
        image_height, image_width = image.shape[:2]
        logger.info(f"📷 Received frame: {image_width}x{image_height} (Debug: {debug_mode})")

        # Step 2: Run YOLO11x inference with MAXIMUM RECALL settings
        # Use torch.no_grad() to disable gradient computation (saves VRAM)
        inference_start = asyncio.get_event_loop().time()

        NAV_CLASSES = [0, 13, 24, 39, 41, 42, 43, 44, 45, 56, 57, 62, 63, 64, 65, 66, 67, 73] 

        with torch.no_grad():
            results = yolo_model(
                image,
                verbose=False,
                conf=0.25,        # Raised from 0.05. Only report if 25% sure.
                iou=0.45,         # Lowered from 0.6. Aggressively merge duplicate boxes.
                imgsz=1280,       # Keep high res for small objects.
                classes=NAV_CLASSES, # <--- CRITICAL: Ignore everything else (fridges, etc).
                agnostic_nms=True # Prevents a "chair" box from overlapping a "bench" box.
            )

        # Calculate inference time in milliseconds
        inference_end = asyncio.get_event_loop().time()
        inference_ms = (inference_end - inference_start) * 1000

        logger.info(f"⚡ Inference: {inference_ms:.2f}ms")

        # Step 3: Convert to semantic spatial data with clustering
        detection_data = process_detections(results, image_height, image_width)
        objects = detection_data['objects']
        clusters = detection_data['clusters']
        emergency_stop = detection_data['emergency_stop']

        # Step 4: Generate natural language summary for LLM
        summary = generate_summary(objects, clusters)

        # Construct final JSON response (matching specified schema)
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emergency_stop": emergency_stop,
            "summary": summary,
            "objects": objects,
            "inference_ms": round(inference_ms, 2)  # GPU inference time for telemetry
        }

        # Step 5: Add debug visualization if requested
        if debug_mode:
            logger.info("🎨 Generating annotated debug frame...")
            annotated_image = annotate_frame(image, objects, emergency_stop)

            # Encode annotated image back to Base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            debug_base64 = base64.b64encode(buffer).decode('utf-8')
            debug_data_url = f"data:image/jpeg;base64,{debug_base64}"

            # Add to response
            response['debug_frame'] = debug_data_url
            logger.info("✓ Debug frame included in response")

        # Calculate total processing time
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.info(
            f"✓ Total: {processing_time:.1f}ms (Inference: {inference_ms:.1f}ms) | "
            f"Objects: {len(objects)} | Emergency: {emergency_stop}"
        )
        logger.info(f"📝 Natural Language Summary: {summary}")

        # Step 5.5: Server-Side Debug Recording (Save annotated frames to disk)
        if SAVE_DEBUG_FRAMES:
            try:
                # Generate annotated frame with all detections
                annotated_debug = annotate_frame(image, objects, emergency_stop)

                # Generate unique filename with timestamp and UUID
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_id = str(uuid.uuid4())[:8]
                filename = f"frame_{timestamp}_{unique_id}.jpg"
                filepath = DEBUG_OUTPUT_DIR / filename

                # Save to disk
                cv2.imwrite(str(filepath), annotated_debug)
                logger.info(f"💾 Saved debug frame: {filename}")

            except Exception as disk_error:
                # Don't crash the websocket if disk I/O fails
                logger.error(f"⚠️  Failed to save debug frame: {disk_error}")

        # Step 6: Call Gemini Live API for audio narration
        scene_analysis = {
            "summary": summary,
            "objects": objects,
            "emergency_stop": emergency_stop
        }

        narration_result = await call_gemini(scene_analysis, base64_frame)

        # Send audio narration to client if available
        if narration_result:
            # Play audio on server speakers for testing (blocking call)
            play_audio_on_server(narration_result.audio_data, sample_rate=24000)

            # Convert audio data to base64 for transmission
            audio_base64 = base64.b64encode(narration_result.audio_data).decode('utf-8')

            await sio.emit('audio_narration', {
                'audio_data': audio_base64,  # Base64-encoded PCM audio (24kHz, 16-bit, mono)
                'transcript': narration_result.transcript,
                'latency_ms': narration_result.latency_ms,
                'total_ms': narration_result.total_ms,
                'cached': narration_result.cached,
                'emergency': emergency_stop,
                'timestamp': response['timestamp']
            }, room=sid)
            logger.info("🔊 Audio narration sent to client")

    except Exception as e:
        logger.error(f"Error processing video frame: {e}", exc_info=True)
        await sio.emit('error', {
            'message': 'Frame processing failed',
            'error': str(e)
        }, room=sid)


if __name__ == "__main__":
    logger.info("🚀 Starting YOLO11 Vision Server...")
    logger.info("Model: YOLO11 Nano (yolo11n.pt)")
    logger.info("Server will run on http://0.0.0.0:8000")

    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
