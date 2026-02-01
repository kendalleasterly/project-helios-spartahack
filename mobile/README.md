# a-eye mobile

Assistive navigation application for blind users using computer vision and AI.

## Acknowledgments

The Kokoro TTS integration in this project is adapted from [expo-kokoro-onnx](https://github.com/isaiahbjork/expo-kokoro-onnx) by Isaiah Bjork (MIT License). We are grateful for this excellent implementation of on-device text-to-speech using ONNX Runtime.

## Setup

1. Install dependencies

   ```bash
   npm install
   ```

2. Configure backend server URL

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and update `BACKEND_SERVER_URL` with your laptop's IP address.

3. Start the app

   ```bash
   npm start
   ```

## Local module wiring (expo-stream-audio)

This app uses a local copy of `expo-stream-audio`.

- Dependency is pinned to the local folder: `"expo-stream-audio": "file:./expo-stream-audio"` in `package.json`.
- JS updates inside `expo-stream-audio/src` require building the module:
  ```bash
  npm run module:build
  ```
- Native updates inside `expo-stream-audio/ios` or `expo-stream-audio/android` require a dev-client rebuild:
  ```bash
  npm run build:dev
  ```
- If Metro is sticky, clean caches:
  ```bash
  npm run metro:clean
  ```

## Common commands

```bash
npm install
npm run typecheck
npm run module:build
npm run build:dev
npm run metro:clean
```
