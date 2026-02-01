# Project Helios 

> **An AI-powered assistive navigation system for blind users using computer vision, real-time object detection, and conversational AI.**

Project Helios (branded as "a-eye") is a comprehensive mobile application that provides real-time spatial awareness and navigation assistance for blind users through advanced computer vision, YOLO11 object detection, and Google's Gemini AI.

## 🌟 Overview

Project Helios combines:
- **Real-time Visual Processing**: YOLO11-based object detection with spatial awareness
- **Conversational AI**: Natural language interaction via Google Gemini
- **Proactive Guidance**: Intelligent heuristics-driven navigation assistance
- **Wake Word Activation**: Hands-free "Helios" wake word detection
- **Mobile-First Design**: Native iOS app built with React Native and Expo

### Key Features

- 🎯 **Real-time Object Detection**: YOLO11 Nano for optimal accuracy/speed balance
- 📍 **Spatial Awareness**: Semantic positioning (Left/Center/Right) and distance estimation
- 🚨 **Emergency Detection**: Automatic hazard detection with haptic feedback
- 🗣️ **Voice Interaction**: Continuous speech recognition with wake word detection
- 🤖 **AI Assistant**: Contextual, conversational responses powered by Gemini
- 📱 **Mobile App**: Cross-platform support for iOS

## 🏗️ Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                      MOBILE APP (iOS)                        │
│  • Camera capture (React Native Vision Camera)              │
│  • Continuous speech recognition                            │
│  • Socket.IO client for real-time communication             │
│  • Text-to-speech output                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket (Socket.IO)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               BACKEND SERVER (Python/FastAPI)                │
│  • YOLO11 object detection                                  │
│  • Heuristics engine (when to speak)                        │
│  • Wake word detection                                      │
│  • Scene history tracking                                   │
└────────────────────────┬────────────────────────────────────┘
                         │ API Calls
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GOOGLE GEMINI AI                          │
│  • Vision mode: Proactive navigation guidance               │
│  • Conversation mode: Answer user questions                 │
│  • Context-aware responses                                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Visual Input**: iPhone camera captures frames at ~1 FPS
2. **Detection**: YOLO11 processes frames and identifies objects with spatial data
3. **Decision**: Heuristics engine determines if guidance is needed
4. **Intelligence**: Gemini AI generates contextual, conversational responses
5. **Output**: Text-to-speech provides audio guidance to the user

### Dual-Pipeline Architecture

**Vision Pipeline** (Proactive):
- Monitors environment continuously
- Speaks when obstacles or hazards detected
- Uses heuristics to avoid over-speaking

**Conversation Pipeline** (Reactive):
- Activated by "Helios" wake word
- Answers user questions
- Provides detailed environmental descriptions

## 🚀 Getting Started

### Prerequisites

- **Backend**: Python 3.10+, pip
- **Mobile**: Node.js 18+, npm, Expo CLI
- **iOS Development**: macOS with Xcode (for iOS builds)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kendalleasterly/project-helios-spartahack.git
   cd project-helios-spartahack
   ```

2. **Set up the backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python server.py
   ```
   
   See [backend/README.md](backend/README.md) for detailed setup instructions.

3. **Set up the mobile app**:
   ```bash
   cd mobile
   npm install
   cp .env.example .env
   # Edit .env and update BACKEND_SERVER_URL with your server IP
   npm start
   ```
   
   See [mobile/README.md](mobile/README.md) for detailed setup instructions.

### API Configuration

You'll need to configure:
- **Google Gemini API Key**: For AI-powered responses
- **Deepgram API Key** (optional): For enhanced speech recognition
- **Backend Server URL**: In mobile app's `.env` file

## 📚 Documentation

Each component has detailed documentation:

- **[Backend Documentation](backend/README.md)**: YOLO11 setup, API endpoints, spatial logic
- **[Mobile Documentation](mobile/README.md)**: React Native setup, module building, commands
- **[Gemini Architecture](backend/GEMINI.md)**: AI decision flow, prompts, personality design
- **[Wake Word API](backend/WAKE_WORD_API.md)**: Voice interaction implementation

## 🎨 Project Structure

```
project-helios-spartahack/
├── backend/                    # Python FastAPI server
│   ├── server.py              # Main server with Socket.IO
│   ├── heuristics.py          # Decision engine for when to speak
│   ├── gemini_service.py      # Gemini API integration
│   ├── contextual_gemini_service.py  # Context-aware Gemini calls
│   └── requirements.txt       # Python dependencies
│
├── mobile/                    # React Native mobile app
│   ├── app/                   # Expo Router pages
│   ├── components/            # React components
│   ├── expo-stream-audio/     # Custom audio streaming module
│   ├── hooks/                 # React hooks
│   └── package.json           # Node.js dependencies
│
└── README.md                  # This file
```

## 🔬 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Socket.IO**: Real-time bidirectional communication
- **YOLO11**: State-of-the-art object detection
- **OpenCV**: Image processing
- **Google Gemini 2.5 Flash**: Conversational AI

### Mobile
- **React Native**: Cross-platform mobile framework
- **Expo**: Development and build tooling
- **React Native Vision Camera**: Camera access
- **@react-native-voice/voice**: Speech recognition
- **Socket.IO Client**: Real-time server communication

## 🎯 Key Innovations

### Heuristics-Driven Guidance
Instead of having the AI decide when to speak, we use deterministic heuristics based on YOLO detection data:
- **Emergency**: Immediate response for vehicles/hazards
- **Alert**: Objects within 3 feet
- **Guidance**: Obstacles in walking path
- **Info**: New important objects detected

This approach provides:
- ✅ Faster response times (no AI decision latency)
- ✅ More reliable behavior (deterministic logic)
- ✅ Reduced API costs (fewer unnecessary calls)
- ✅ Better user experience (predictable assistance)

### Helios Personality
The AI assistant is designed with a distinct personality:
- **Warm but not patronizing**: Helpful friend, not a robot
- **Direct but not robotic**: Gets to the point naturally
- **Calm in emergencies**: Steady guidance under pressure
- **Honest about uncertainty**: Admits when it can't see clearly

Example interactions:
- Emergency: "Stop! Car left!"
- Guidance: "Chair ahead, veer right."
- Conversation: "Yeah, there's a door about 10 feet ahead."

## 🧪 Development

### Running Tests
```bash
# Backend tests (if available)
cd backend
python -m pytest

# Mobile type checking
cd mobile
npm run typecheck
```

### Building the App
```bash
# iOS development build
cd mobile
npm run build:dev

# Clean cache if needed
npm run metro:clean
```

### Local Module Development
The mobile app includes a custom `expo-stream-audio` module:
```bash
# After JS changes in expo-stream-audio/src
npm run module:build

# After native changes in expo-stream-audio/ios
npm run build:dev
```

## 🤝 Contributing

This project was developed for SpartaHack 9. Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

MIT License - See individual component documentation for details.

## 🙏 Acknowledgments

- **SpartaHack 9**: For providing the opportunity to build this project
- **Ultralytics**: For the YOLO11 object detection model
- **Google**: For Gemini AI API access
- **Expo Team**: For excellent React Native tooling

## 📞 Support

For questions or issues:
- Check component-specific README files
- Review backend logs with DEBUG level enabled
- Open an issue on GitHub

---

**Built with ❤️ for accessibility and inclusion**
