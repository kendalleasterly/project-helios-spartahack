# YOLO11 Vision Processing Backend

Python FastAPI + Socket.IO backend for the **Reflex Layer** of a wearable AI assistant for blind people. Receives video frames from an iPhone, processes them with YOLO11, and outputs semantic spatial data for LLM narration.

## Architecture

```
iPhone (Socket.IO Client)
    ↓ Base64 JPEG frames
Python Server (FastAPI + Socket.IO)
    ↓ Decode & YOLO11 Inference
Semantic Spatial Data (JSON)
    ↓ Summary string
Gemini LLM → Audio Narration
```

## Features

- **Real-time Object Detection**: YOLO11 Nano for optimal accuracy/speed balance
- **Spatial Awareness**: Converts bounding boxes to semantic positions (Left/Center/Right) and distances (Immediate/Close/Far)
- **Emergency Detection**: Automatically flags dangerous vehicles nearby for haptic feedback
- **LLM-Ready Output**: Structured JSON with summary text for easy Gemini integration

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The first time you run the server, YOLO11 model weights (`yolo11n.pt`) will be automatically downloaded by ultralytics (~6MB).

## Running the Server

```bash
python server.py
```

The server will start on `http://0.0.0.0:8000`

### Health Check

Visit [http://localhost:8000/health](http://localhost:8000/health) to verify the server is running and the model is loaded.

## API Endpoints

### HTTP Endpoints

- `GET /` - Basic status check
- `GET /health` - Detailed health status with model info

### Socket.IO Events

#### Client → Server

**Event**: `video_frame`

**Payload**:
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

#### Server → Client

**Event**: `scene_analysis`

**Payload** (matches schema for Gemini team):
```json
{
  "timestamp": "2026-01-31T12:34:56.789Z",
  "emergency_stop": false,
  "summary": "Person (Center, Close), Chair (Left, Far)",
  "objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "position": "center",
      "distance": "close",
      "box": [120, 80, 320, 450]
    },
    {
      "label": "chair",
      "confidence": 0.87,
      "position": "left",
      "distance": "far",
      "box": [10, 200, 100, 380]
    }
  ]
}
```

## Spatial Logic Explained

### Horizontal Position
Based on bounding box center X-coordinate:
- **Left**: 0-33% of image width
- **Center**: 33-66% of image width
- **Right**: 66-100% of image width

### Distance Estimation
Based on bounding box height relative to image:
- **Immediate**: Box height > 50% of image (object is RIGHT IN FRONT)
- **Close**: Box height > 20% of image
- **Far**: Box height ≤ 20% of image

### Emergency Detection

`emergency_stop` flag is set to `true` when:
- Detected object is a **vehicle** (car, bus, truck, motorcycle, bicycle)
- **AND** distance is **immediate** or **close**

This allows the iOS client to trigger haptic feedback **immediately** without waiting for LLM processing.

## Integration with Gemini

The JSON output is designed for easy LLM integration:

1. **emergency_stop**: Check client-side for instant haptic buzz
2. **summary**: Inject directly into Gemini prompt:
   ```
   Context: Person (Center, Close), Car (Right, Immediate)
   Task: Generate warning for blind user
   ```
3. **objects array**: Full structured data for advanced reasoning
4. **distance field**: Provides urgency context ("immediate" vs "close")

## Why YOLO11?

We chose **YOLO11 over YOLOv8** because it offers:
- **Superior feature extraction accuracy** (especially for small/distant objects)
- **Same ultralytics API** for easy integration
- **Marginal speed difference** at Nano scale (~5ms)
- **Better generalization** to varied lighting conditions

The Nano variant (`yolo11n.pt`) provides the best balance for real-time mobile applications.

## Development

### Testing Socket.IO Connection

You can test the server with a simple Python client:

```python
import socketio
import base64

sio = socketio.Client()

@sio.on('scene_analysis')
def on_analysis(data):
    print('Received:', data)

sio.connect('http://localhost:8000')

# Send test frame
with open('test_image.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
    sio.emit('video_frame', {'frame': f'data:image/jpeg;base64,{b64}'})

sio.wait()
```

### Logging

The server logs processing time for each frame:
```
✓ Processed in 45.2ms | Objects: 3 | Emergency: False
```

Monitor for performance issues or emergency detections.

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-socketio` - WebSocket support
- `ultralytics` - YOLO11 implementation
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pillow` - Image decoding

## Troubleshooting

### Model download fails
If the YOLO11 model doesn't auto-download, manually download `yolo11n.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in the backend directory.

### CORS errors
The server is configured with `cors_allowed_origins='*'` for development. For production, update this in [server.py:32](server.py#L32) to your specific iPhone client origin.

### Performance issues
- Ensure you're using YOLO11 **Nano** (not Small/Medium)
- Check CPU/GPU usage
- Consider reducing frame rate on iPhone side

## License

MIT
