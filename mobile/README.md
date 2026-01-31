# a-eye mobile

## Local module wiring (expo-stream-audio)

This app uses a local copy of `expo-stream-audio`.

- Dependency is pinned to the local folder: `"expo-stream-audio": "file:./expo-stream-audio"` in `package.json`.
- JS updates inside `expo-stream-audio/src` require building the module:
  ```bash
  npm run module:build
  ```
- Native updates inside `expo-stream-audio/ios` or `expo-stream-audio/android` require a dev-client rebuild:
  ```bash
  npm run build:dev:local
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
npm run build:dev:local
npm run metro:clean
```
