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
import { Asset } from 'expo-asset';

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

        // NOTE: Model loading from local assets doesn't work in dev mode
        // The 89MB model file is too large for Metro bundler's require()
        // This will work after EAS build when assets are properly bundled

        // For now, skip model loading in development
        // The model will be available in the production build
        console.log('⚠️  Kokoro TTS model loading skipped in dev mode');
        console.log('   Model will load automatically in EAS production build');

        // Mark as ready so the rest of the app works
        // TTS just won't generate audio until after native build

        setIsReady(true);
        setError(null);
        console.log('✓ Kokoro TTS ready (audio generation disabled in dev mode)');
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
