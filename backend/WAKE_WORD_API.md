# Wake Word Detection API Documentation

## Overview
The backend now supports continuous audio transcription with **wake word detection**. The system listens for the wake word **"helios"** and accumulates the user's question, responding after 1.5 seconds of silence.

## Wake Word Variations
The system recognizes multiple spellings to handle transcription errors:
- `helios`
- `helius`
- `helias`
- `heleios`
- `heleos`
- `hellios`
- `hey helios`
- `ok helios`

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User speaks: "Helios, where is my laptop?"                  │
│    ↓                                                            │
│ 2. Frontend transcribes continuously                           │
│    ↓                                                            │
│ 3. Backend detects "helios" → activates listening              │
│    ↓                                                            │
│ 4. Backend accumulates: "where is my laptop?"                  │
│    ↓                                                            │
│ 5. User stops speaking → 1.5s silence                          │
│    ↓                                                            │
│ 6. Backend processes question with Gemini                      │
│    ↓                                                            │
│ 7. Backend responds: "Your laptop is on the table..."          │
└─────────────────────────────────────────────────────────────────┘
```

## Frontend Implementation

### Required: Continuous Transcription

The frontend must continuously transcribe audio and send transcriptions to the backend as the user speaks. **Do NOT wait for complete utterances** - send partial transcriptions.

#### Example: iOS Speech Recognition

```swift
import Speech

class ContinuousTranscription {
    private var recognitionTask: SFSpeechRecognitionTask?
    private let recognizer = SFSpeechRecognizer()

    func startContinuousListening() {
        let request = SFSpeechAudioBufferRecognitionRequest()

        // IMPORTANT: Use partial results
        request.shouldReportPartialResults = true

        recognitionTask = recognizer?.recognitionTask(with: request) { result, error in
            if let result = result {
                // Send EVERY partial transcription to backend
                let transcription = result.bestTranscription.formattedString
                self.sendToBackend(transcription)
            }
        }

        // Feed audio to recognition request
        audioEngine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }
    }

    func sendToBackend(_ text: String) {
        socket.emit('video_frame_streaming', [
            'frame': getCurrentFrame(),
            'user_question': text  // Send continuous transcription
        ])
    }
}
```

### WebSocket Data Format

#### Send Continuous Transcriptions

```javascript
// Send transcriptions as they arrive (NOT when user stops speaking)
speechRecognizer.on('partialResult', (text) => {
  socket.emit('video_frame_streaming', {
    frame: getCurrentFrame(),           // Current camera frame
    user_question: text                 // Partial transcription
  });
});

// Example sequence:
// User: "Helios, where is my laptop?"
//
// Backend receives:
// 1. user_question: "helios"
// 2. user_question: "helios where"
// 3. user_question: "helios where is"
// 4. user_question: "helios where is my"
// 5. user_question: "helios where is my laptop"
// 6. [1.5s silence]
// 7. Backend processes: "where is my laptop"
```

#### Receive Response

```javascript
socket.on('text_response', (data) => {
  console.log('Response:', data.text);
  // data = {
  //   text: "Your laptop is on the table to your right.",
  //   mode: "conversation",
  //   emergency: false
  // }

  speakToUser(data.text);
});
```

## Backend Behavior

### State Machine

The backend maintains a state machine per client:

| State | Description | Behavior |
|-------|-------------|----------|
| **Inactive** | No wake word detected | Ignores all transcriptions, logs at DEBUG level |
| **Active - Accumulating** | Wake word detected, listening for question | Accumulates text, resets silence timer on each new transcription |
| **Processing** | Silence timeout reached | Sends accumulated question to Gemini, responds to user |

### Timing

- **Silence timeout**: 1500ms (1.5 seconds)
- **Timer reset**: Every time new transcription arrives
- **Processing time**: 3-5 seconds (Gemini API call)

### Example Log Output

```
# User says: "Helios, where is my laptop?"

📷 Frame [VISION]: 720x1280 | Lock: 🔓 UNLOCKED | Question: None
🎤 Transcription received: 'helios'
🎤 Wake word detected: 'helios' → Question: ''
🎤 WAKE WORD ACTIVATED - listening for question...

🎤 Transcription received: 'helios where'
🎤 Accumulating: 'where' → Total: 'where'

🎤 Transcription received: 'helios where is my'
🎤 Accumulating: 'where is my' → Total: 'where is my'

🎤 Transcription received: 'helios where is my laptop'
🎤 Accumulating: 'where is my laptop' → Total: 'where is my laptop'

[1.5 seconds of silence]

⏱️  Silence timeout reached (1500ms) - processing question
🤖 Processing wake word question: 'where is my laptop'
🤖 Calling Gemini API with wake word question...
✅ Gemini response (3245ms): Your laptop is on the table to your right, about 3 feet away.
```

## Important Notes

### ✅ DO

1. **Send continuous transcriptions**: Don't wait for user to finish speaking
2. **Send with every frame**: Include current camera frame with each transcription
3. **Use partial results**: Enable partial transcription results in speech recognizer
4. **Send duplicates**: It's OK to send the same text multiple times, backend handles deduplication
5. **Let backend handle silence**: Backend automatically detects when user stops speaking

### ❌ DON'T

1. **Don't wait for complete sentences**: Send as soon as you have text
2. **Don't do silence detection on frontend**: Backend handles this
3. **Don't filter wake words on frontend**: Backend does this
4. **Don't debounce transcriptions**: Send immediately as they arrive
5. **Don't send empty strings**: Only send when you have actual text

## Testing

### Test Case 1: Basic Wake Word
```
User: "Helios, what do you see?"
Expected: Backend responds with scene description
```

### Test Case 2: Wake Word Spelling Variation
```
User: "Helius, where am I?"  (misspelled)
Expected: Backend still recognizes and responds
```

### Test Case 3: No Wake Word
```
User: "I'm looking for my laptop"  (no wake word)
Expected: Backend ignores, no response
```

### Test Case 4: Wake Word Mid-Sentence
```
User: "Hey Helios, can you help me?"
Expected: Backend extracts "can you help me?" and responds
```

### Test Case 5: Multiple Questions
```
User: "Helios, what's on the table?"
[Wait for response]
User: "Helios, where's the door?"
Expected: Two separate responses
```

## Troubleshooting

### Problem: Backend doesn't respond
**Cause**: Wake word not detected
**Solution**:
- Check logs for "No wake word detected"
- Verify transcription includes one of the wake word variations
- Try saying "hey helios" or "ok helios"

### Problem: Partial responses
**Cause**: Silence timeout too short, user still speaking
**Solution**: Speak more quickly or ask backend team to increase `SILENCE_TIMEOUT_MS`

### Problem: Delayed responses
**Cause**: Normal - Gemini API takes 3-5 seconds
**Solution**: Show loading indicator to user during processing

### Problem: Backend processes before user finishes
**Cause**: 1.5s pause during question
**Solution**: Speak without pauses, or ask backend team to increase timeout

## Configuration

Backend configuration (in `server.py`):

```python
SILENCE_TIMEOUT_MS = 1500  # Increase if users need more time between words
WAKE_WORD_VARIATIONS = [    # Add more variations if needed
    'helios',
    'helius',
    # ...
]
```

## Complete Example

```javascript
// iOS/React Native Example

import { socket } from './socket';
import Voice from '@react-native-voice/voice';

class VoiceAssistant {
  constructor() {
    this.isListening = false;

    // Setup voice recognition
    Voice.onSpeechResults = this.onSpeechResults.bind(this);
    Voice.onSpeechPartialResults = this.onSpeechPartialResults.bind(this);
  }

  async start() {
    this.isListening = true;
    await Voice.start('en-US');
  }

  async stop() {
    this.isListening = false;
    await Voice.stop();
  }

  // IMPORTANT: Use partial results, not final results
  onSpeechPartialResults(event) {
    if (!this.isListening) return;

    const transcription = event.value[0];

    // Send to backend immediately
    socket.emit('video_frame_streaming', {
      frame: this.getCurrentFrame(),
      user_question: transcription
    });
  }

  onSpeechResults(event) {
    // Final result - also send it
    const transcription = event.value[0];

    socket.emit('video_frame_streaming', {
      frame: this.getCurrentFrame(),
      user_question: transcription
    });
  }

  getCurrentFrame() {
    // Return base64 encoded camera frame
    return this.cameraFrameBase64;
  }
}

// Usage
const assistant = new VoiceAssistant();

// Start continuous listening when app launches
assistant.start();

// Listen for responses
socket.on('text_response', ({ text, mode, emergency }) => {
  if (emergency) {
    playUrgentSound();
  }

  // Convert text to speech and play
  speakText(text);
});
```

## Summary

| What | Where | When |
|------|-------|------|
| **Wake Word Detection** | Backend | Checks every transcription for "helios" |
| **Question Accumulation** | Backend | After wake word detected |
| **Silence Detection** | Backend | 1.5s after last transcription |
| **Speech-to-Text** | Frontend | Continuous, partial results |
| **Gemini Processing** | Backend | After silence timeout |
| **Text-to-Speech** | Frontend | When response received |

---

**Questions?** Contact the backend team or check server logs with DEBUG level enabled.
