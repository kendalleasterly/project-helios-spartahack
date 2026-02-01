"""
YOLO11 Vision Processing Server for Blind Assistant
Receives video frames via Socket.IO, processes with YOLO11, returns semantic spatial data
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import cv2
import networkx as nx
import numpy as np
import socketio
try:
    import sounddevice as sd
except Exception:  # Optional dependency for server-side audio playback
    sd = None
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from PIL import Image
from ultralytics import YOLO

from contextual_gemini_service import BlindAssistantService, ContextConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global YOLO model (loaded at startup)
yolo_model = None

# Global Blind Assistant Service (dual-pipeline architecture)
assistant = None

# Lock to ensure only one Gemini vision call at a time
gemini_vision_lock = asyncio.Lock()

# Vehicle classes that trigger emergency warnings
VEHICLE_CLASSES = {'car', 'bus', 'truck', 'motorcycle', 'bicycle'}

# Emergency distance thresholds
EMERGENCY_DISTANCES = {'immediate', 'close'}

# Debug Recording Configuration
SAVE_DEBUG_FRAMES = os.getenv('SAVE_DEBUG_FRAMES', 'true').lower() == 'true'
DEBUG_OUTPUT_DIR = Path('server_debug_output')
DEVICE_STREAM_LOG_MAX_CHARS = int(os.getenv('DEVICE_STREAM_LOG_MAX_CHARS', '1200'))

# Track device stream message counts per socket
device_stream_counts: Dict[str, int] = {}

# Store latest sensor data per socket
device_sensor_cache: Dict[str, Dict[str, Any]] = {}

# Performance tracking
inference_times = []
MAX_TIMING_HISTORY = 30

def _select_torch_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_yolo_model():
    """Load YOLO11 Small model."""
    global yolo_model
    logger.info("Loading YOLO11 Small model...")
    try:
        device = _select_torch_device()
        # Switching to yolo11s.pt for faster inference (better FPS)
        yolo_model = YOLO('yolo11s.pt')
        yolo_model.to(device)
        logger.info(f"✓ YOLO11s loaded successfully on {device.upper()}")
    except Exception as e:
        logger.error(f"✗ Failed to load YOLO11 model: {e}")
        raise


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode Base64 JPEG to OpenCV BGR."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise


def calculate_oval_position(bbox: List[int], image_width: int, image_height: int) -> str:
    """
    Determine if an object is in the user's immediate path using an oval region.
    Checks if ANY part of the bounding box intersects with the oval.
    """
    x1, y1, x2, y2 = bbox
    
    # Ignore objects ONLY if they are 100% inside the blind spot (e.g. feet)
    if y1 > image_height * 0.9:
        return "peripheral"
    
    # Define oval path region
    oval_center_x = image_width / 2
    # Shifted to 1.3 (middle ground)
    oval_center_y = image_height * 1.3
    radius_x = image_width * 0.28
    radius_y = image_height * 0.6
    
    # Points to check: Corners, Midpoints, Center
    points = [
        (x1, y1), (x2, y1), (x1, y2), (x2, y2), # Corners
        ((x1+x2)/2, y1), ((x1+x2)/2, y2),       # Top/Bottom mid
        (x1, (y1+y2)/2), (x2, (y1+y2)/2),       # Left/Right mid
        ((x1+x2)/2, (y1+y2)/2)                  # Center
    ]
    
    # Check if ANY point is inside the oval equation
    # ((x - h)^2 / rx^2) + ((y - k)^2 / ry^2) <= 1
    for px, py in points:
        normalized_dist = (((px - oval_center_x)**2) / (radius_x**2)) + \
                          (((py - oval_center_y)**2) / (radius_y**2))
        if normalized_dist <= 1.0:
            return "path"
    
    return "peripheral"


OBJECT_HEIGHTS = {
    'person': 1.7, 'bench': 0.8, 'backpack': 0.45, 'chair': 0.9, 'couch': 0.85,
    'bottle': 0.25, 'cup': 0.12, 'fork': 0.18, 'knife': 0.20, 'spoon': 0.18, 'bowl': 0.08,
    'tv': 0.80, 'laptop': 0.02, 'mouse': 0.04, 'remote': 0.18, 'keyboard': 0.03, 'cell phone': 0.15, 'book': 0.23,
    'car': 1.5, 'truck': 2.5, 'bus': 3.0, 'motorcycle': 1.2, 'bicycle': 1.1, 'default': 1.0
}

IPHONE_FOCAL_LENGTH_MM = 6.86
IPHONE_SENSOR_HEIGHT_MM = 7.3
IPHONE_IMAGE_HEIGHT_PX = 720
FOCAL_LENGTH_PIXELS = (IPHONE_FOCAL_LENGTH_MM * IPHONE_IMAGE_HEIGHT_PX) / IPHONE_SENSOR_HEIGHT_MM


def calculate_distance(bbox_height: float, image_height: int, label: str = 'default') -> str:
    """Advanced distance estimation."""
    if bbox_height <= 0: return "unknown"
    real_height_m = OBJECT_HEIGHTS.get(label, OBJECT_HEIGHTS['default'])
    focal_length_adjusted = FOCAL_LENGTH_PIXELS * (image_height / IPHONE_IMAGE_HEIGHT_PX)
    distance_meters = (real_height_m * focal_length_adjusted) / bbox_height
    distance_feet = distance_meters * 3.28084
    if distance_feet < 3: category = "immediate"
    elif distance_feet < 8: category = "close"
    elif distance_feet < 15: category = "medium"
    else: category = "far"
    return f"{category} ({distance_feet:.1f} ft)"


def annotate_frame(image: np.ndarray, objects: List[Dict[str, Any]],
                   emergency_stop: bool = False) -> np.ndarray:
    """Annotate image for debugging."""
    annotated = image.copy()
    h, w = image.shape[:2]
    
    # Draw path oval for visualization (shifted)
    cv2.ellipse(annotated, (int(w/2), int(h*1.3)), (int(w*0.28), int(h*0.6)), 0, 180, 360, (255, 255, 0), 2)
    # Draw excluded bottom region line
    cv2.line(annotated, (0, int(h*0.9)), (w, int(h*0.9)), (0, 0, 255), 2)
    
    if emergency_stop:
        cv2.rectangle(annotated, (0, 0), (image.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(annotated, "EMERGENCY: VEHICLE DETECTED", (10, 35), cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)
    
    for obj in objects:
        label = obj.get('label', 'unknown')
        confidence = obj.get('confidence', 0.0)
        position = obj.get('position', 'unknown')
        distance = obj.get('distance', 'unknown')
        x1, y1, x2, y2 = obj.get('box', [0, 0, 0, 0])
        dist_cat = distance.split()[0] if ' ' in distance else distance
        
        # Red if immediate path hazard
        is_hazard = position == "path" and (dist_cat == 'immediate' or dist_cat == 'close')
        color = (0, 0, 255) if is_hazard else (0, 255, 0)
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {int(confidence*100)}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def calculate_centroid(obj: Dict[str, Any]) -> tuple:
    x1, y1, x2, y2 = obj['box']
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(point1: tuple, point2: tuple) -> float:
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def detect_crowd_clusters(objects: List[Dict[str, Any]], proximity_threshold: float = 200.0, image_width: int = 1280, image_height: int = 720) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups = defaultdict(list)
    for obj in objects: groups[obj['label']].append(obj)
    clusters = []
    for label, group in groups.items():
        if len(group) < 3: continue
        G = nx.Graph()
        for idx, obj in enumerate(group): G.add_node(idx, obj=obj)
        for i, obj_i in enumerate(group):
            c_i = calculate_centroid(obj_i)
            for j, obj_j in enumerate(group):
                if i < j and euclidean_distance(c_i, calculate_centroid(obj_j)) < proximity_threshold: G.add_edge(i, j)
        for comp in nx.connected_components(G):
            if len(comp) < 3: continue
            cluster_objs = [group[idx] for idx in comp]
            boxes = [o['box'] for o in cluster_objs]
            min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
            max_x, max_y = max(b[2] for b in boxes), max(b[3] for b in boxes)
            clusters.append({'type': 'crowd', 'label': label, 'count': len(comp), 'description': f"cluster of {len(comp)} {label}s", 'bounding_box': [min_x, min_y, max_x, max_y]})
    return clusters


def process_detections(results, image_height: int, image_width: int) -> Dict[str, Any]:
    objects = []
    emergency_stop = False
    if results and results[0].boxes is not None:
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            label = results[0].names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            # Use new oval position logic
            pos = calculate_oval_position([int(x1), int(y1), int(x2), int(y2)], image_width, image_height)
            dist = calculate_distance(y2-y1, image_height, label)
            objects.append({"id": idx, "label": label, "confidence": conf, "position": pos, "distance": dist, "box": [int(x1), int(y1), int(x2), int(y2)]})
            if label in VEHICLE_CLASSES and dist.split()[0] in EMERGENCY_DISTANCES: emergency_stop = True
    return {"objects": objects, "clusters": detect_crowd_clusters(objects), "emergency_stop": emergency_stop}


def generate_summary(objects: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> str:
    if not objects: return "Clear path ahead."
    parts = [c['description'] for c in clusters]
    clustered_labels = {c['label'] for c in clusters}
    for obj in objects:
        if obj['label'] not in clustered_labels: parts.append(f"a {obj['label']} ({obj['position']})")
    return ", ".join(parts).capitalize() + "."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    load_yolo_model()
    try:
        assistant = BlindAssistantService(ContextConfig(spatial_lookback_seconds=30, max_scene_history=60, store_frames=False))
        logger.info("✓ Blind Assistant Service initialized")
    except Exception as e:
        logger.error(f"✗ Assistant initialization failed: {e}")
    yield
    logger.info("🛑 Shutting down server...")


app = FastAPI(title="YOLO11 Vision Server", lifespan=lifespan)
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)


@sio.event
async def connect(sid, environ):
    logger.info(f"✓ Client connected: {sid}")
    device_stream_counts[sid] = 0


@sio.event
async def disconnect(sid):
    logger.info(f"✗ Client disconnected: {sid}")
    device_sensor_cache.pop(sid, None)


@sio.event
async def device_sensor_stream(sid, data):
    device_stream_counts[sid] = device_stream_counts.get(sid, 0) + 1
    device_sensor_cache[sid] = {
        "speed_mps": data.get("speed_mps"),
        "speed_avg_1s_mps": data.get("speed_avg_1s_mps"),
        "velocity_x_mps": data.get("velocity_x_mps"),
        "velocity_z_mps": data.get("velocity_z_mps"),
        "magnetic_x_ut": data.get("magnetic_x_ut"),
        "magnetic_z_ut": data.get("magnetic_z_ut"),
        "steps_last_3s": data.get("steps_last_3s"),
        "steps_since_open": data.get("steps_since_open"),
    }


@sio.event
async def video_frame_streaming(sid, data):
    try:
        start_time = asyncio.get_event_loop().time()
        base64_frame = data['frame']
        user_question = data.get('user_question')
        debug_mode = data.get('debug', False)

        image = decode_base64_image(base64_frame)
        h, w = image.shape[:2]

        # Timing ONLY the model inference
        yolo_start = asyncio.get_event_loop().time()
        with torch.no_grad():
            results = yolo_model(image, verbose=False, conf=0.25, iou=0.45, imgsz=1280, classes=[0, 13, 24, 39, 41, 42, 43, 44, 45, 56, 57, 62, 63, 64, 65, 66, 67, 73], agnostic_nms=True)
        yolo_end = asyncio.get_event_loop().time()
        
        inference_ms = (yolo_end - yolo_start) * 1000
        
        # Update rolling average
        inference_times.append(inference_ms)
        if len(inference_times) > MAX_TIMING_HISTORY:
            inference_times.pop(0)
        
        avg_inference_ms = sum(inference_times) / len(inference_times)
        potential_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
        
        logger.info(f"⚡ YOLO: {inference_ms:.1f}ms (Avg: {avg_inference_ms:.1f}ms | Max FPS: {potential_fps:.1f})")

        det = process_detections(results, h, w)
        objects, clusters, emergency_stop = det['objects'], det['clusters'], det['emergency_stop']
        summary = generate_summary(objects, clusters)

        sensor = device_sensor_cache.get(sid, {})
        speed = sensor.get("speed_mps")
        steps = sensor.get("steps_last_3s")
        is_moving = (speed >= 0.2 if speed is not None else (steps > 0 if steps is not None else True))

        yolo_results = {
            "summary": summary, "objects": objects, "emergency_stop": emergency_stop,
            "motion": {**sensor, "is_moving": is_moving}
        }

        if user_question:
            async with gemini_vision_lock:
                resp = await assistant.process_user_speech(user_question)
                await sio.emit('text_response', {'text': resp, 'mode': 'conversation', 'emergency': False}, room=sid)
        else:
            resp = await assistant.process_frame(base64_frame, yolo_results)
            if resp:
                await sio.emit('text_response', {'text': resp, 'mode': 'vision', 'emergency': "STOP" in resp}, room=sid)
                
                # Save debug frame ONLY if a warning/response was triggered
                if SAVE_DEBUG_FRAMES:
                    try:
                        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        filename = f"{DEBUG_OUTPUT_DIR}/alert_{timestamp}.jpg"
                        
                        # Use the annotated frame for debugging
                        ann = annotate_frame(image, objects, emergency_stop)
                        
                        # Add motion info to the saved image
                        cv2.putText(ann, f"Speed: {speed:.2f} m/s | Moving: {is_moving}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(ann, f"Reason: {resp}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        cv2.imwrite(filename, ann)
                        logger.info(f"💾 Saved alert debug frame: {filename}")
                    except Exception as save_err:
                        logger.error(f"Failed to save debug frame: {save_err}")

        # Emit raw detection data for frontend visualization
        await sio.emit('detection_update', {
            'objects': objects,
            'motion': yolo_results['motion'],
            'summary': summary,
            'is_moving': is_moving,
            'emergency': emergency_stop,
            'performance': {
                'curr_ms': round(inference_ms, 1),
                'avg_ms': round(avg_inference_ms, 1),
                'fps': round(potential_fps, 1)
            }
        }, room=sid)

        if debug_mode:
            ann = annotate_frame(image, objects, emergency_stop)
            _, buf = cv2.imencode('.jpg', ann)
            await sio.emit('debug_frame', {'frame': f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", 'summary': summary, 'mode': 'vision'}, room=sid)

    except Exception as e:
        logger.error(f"Error: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def video_frame(sid, data):
    # Minimal implementation for legacy
    pass


if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, log_level="info")