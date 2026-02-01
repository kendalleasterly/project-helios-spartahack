/**
 * Audio Guidance Hook
 *
 * Orchestrates the complete TTS pipeline:
 * 1. Listen to text tokens from WebSocket
 * 2. Buffer tokens until sentence boundary
 * 3. Trigger Kokoro TTS
 * 4. Play audio
 */

import { useEffect, useRef, useState } from 'react';
import { useTextBuffer } from './useTextBuffer';
import type { TextTokenEvent } from './useWebSocket';
import * as FileSystem from 'expo-file-system/legacy';

// Lazy import Kokoro to avoid loading on module import
let kokoroInstance: any = null;

const MODEL_ID = 'quantized';
const VOICE_ID = 'af_bella';

interface UseAudioGuidanceOptions {
  onTextToken: (callback: (event: TextTokenEvent) => void) => void;
  enabled?: boolean;
}

export function useAudioGuidance({ onTextToken, enabled = true }: UseAudioGuidanceOptions) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const textBuffer = useTextBuffer();
  const audioQueueRef = useRef<any[]>([]);
  const isPlayingRef = useRef(false);

  // Initialize Kokoro TTS
  useEffect(() => {
    if (!enabled || isReady || isInitializing) return;

    async function initKokoro() {
      try {
        setIsInitializing(true);
        console.log('Initializing Kokoro TTS...');

        // Lazy load Kokoro singleton instance
        if (!kokoroInstance) {
          const module = await import('@/services/kokoro/kokoroOnnx');
          kokoroInstance = module.default;
        }

        // Download the model from HuggingFace if not cached
        const modelPath = `${FileSystem.cacheDirectory}${MODEL_ID}.onnx`;
        const modelInfo = await FileSystem.getInfoAsync(modelPath);

        if (!modelInfo.exists) {
          console.log('Downloading Kokoro model (89 MB, first launch only)...');
          const modelUrl = 'https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx';

          const downloadResult = await FileSystem.downloadAsync(modelUrl, modelPath);

          if (downloadResult.status !== 200) {
            throw new Error(`Failed to download model: ${downloadResult.status}`);
          }

          console.log('Model downloaded successfully');
        } else {
          console.log('Model already cached');
        }

        // Load the model into Kokoro
        const loaded = await kokoroInstance.loadModel(`${MODEL_ID}.onnx`);

        if (!loaded) {
          throw new Error('Failed to load model into ONNX runtime');
        }

        setIsReady(true);
        setError(null);
        console.log('✓ Kokoro TTS ready');
      } catch (err) {
        console.error('Failed to initialize Kokoro:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsReady(false);
      } finally {
        setIsInitializing(false);
      }
    }

    initKokoro();
  }, [enabled, isReady, isInitializing]);

  // Play audio from queue
  const playNextInQueue = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);

    const sound = audioQueueRef.current.shift();

    try {
      await sound.playAsync();

      sound.setOnPlaybackStatusUpdate((status: any) => {
        if (status.didJustFinish) {
          sound.unloadAsync();
          isPlayingRef.current = false;
          setIsSpeaking(false);
          playNextInQueue();
        }
      });
    } catch (err) {
      console.error('Audio playback error:', err);
      isPlayingRef.current = false;
      setIsSpeaking(false);
      playNextInQueue();
    }
  };

  // Handle complete sentences
  useEffect(() => {
    textBuffer.onSentence(async ({ sentence, emergency }) => {
      if (!isReady || !kokoroInstance) {
        console.warn('Kokoro not ready, skipping TTS');
        return;
      }

      try {
        console.log(`Generating TTS for: "${sentence}" (emergency: ${emergency})`);

        const sound = await kokoroInstance.generateAudio(sentence, VOICE_ID, 1.0);

        if (emergency) {
          audioQueueRef.current.unshift(sound);
        } else {
          audioQueueRef.current.push(sound);
        }

        playNextInQueue();
      } catch (err) {
        console.error('TTS generation error:', err);
      }
    });
  }, [textBuffer, isReady]);

  // Handle incoming text tokens
  useEffect(() => {
    if (!enabled) return;

    onTextToken((event: TextTokenEvent) => {
      textBuffer.addToken(event.token, event.emergency);
    });
  }, [onTextToken, textBuffer, enabled]);

  return {
    isSpeaking,
    isReady,
    isInitializing,
    error,
  };
}
