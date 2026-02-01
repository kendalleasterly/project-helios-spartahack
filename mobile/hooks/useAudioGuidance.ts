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
import { Audio } from 'expo-av';

// Lazy import Kokoro to avoid loading on module import
let kokoroInstance: unknown = null;
let audioModeConfigured = false;

const MODEL_ID = 'quantized';
const VOICE_ID = 'af_bella';

interface UseAudioGuidanceOptions {
  onTextToken: (callback: (event: TextTokenEvent) => void) => void;
  enabled?: boolean;
}

interface KokoroInstance {
  loadModel: (modelId: string) => Promise<boolean>;
  generateAudio: (text: string, voiceId: string, speed: number) => Promise<AudioSound>;
}

interface AudioSound {
  playAsync: () => Promise<void>;
  setOnPlaybackStatusUpdate: (callback: (status: PlaybackStatus) => void) => void;
  unloadAsync: () => Promise<void>;
}

interface PlaybackStatus {
  didJustFinish?: boolean;
}

export function useAudioGuidance({ onTextToken, enabled = true }: UseAudioGuidanceOptions) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const textBuffer = useTextBuffer();
  const audioQueueRef = useRef<AudioSound[]>([]);
  const isPlayingRef = useRef(false);
  const callbackRegisteredRef = useRef(false);

  // Initialize Kokoro TTS
  useEffect(() => {
    if (!enabled || isReady || isInitializing) return;

    async function initKokoro() {
      try {
        setIsInitializing(true);
        console.log('[AudioGuidance] Initializing Kokoro TTS...');

        // Configure audio session for iOS - MUST be done before playback
        if (!audioModeConfigured) {
          console.log('[AudioGuidance] Configuring audio session...');
          await Audio.setAudioModeAsync({
            allowsRecordingIOS: false,
            playsInSilentModeIOS: true,
            staysActiveInBackground: false,
            shouldDuckAndroid: true,
            playThroughEarpieceAndroid: false,
          });
          audioModeConfigured = true;
          console.log('[AudioGuidance] Audio session configured');
        }

        // Lazy load Kokoro singleton instance
        if (!kokoroInstance) {
          console.log('[AudioGuidance] Importing Kokoro module...');
          const module = await import('@/services/kokoro/kokoroOnnx');
          kokoroInstance = module.default;
          console.log('[AudioGuidance] Kokoro module imported');
        }

        // Download the model from HuggingFace if not cached
        const modelPath = `${FileSystem.cacheDirectory}${MODEL_ID}.onnx`;
        console.log('[AudioGuidance] Checking for model at:', modelPath);
        const modelInfo = await FileSystem.getInfoAsync(modelPath);

        if (!modelInfo.exists) {
          console.log('[AudioGuidance] Downloading Kokoro model (89 MB, first launch only)...');
          const modelUrl = 'https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx';

          const downloadResult = await FileSystem.downloadAsync(modelUrl, modelPath);

          if (downloadResult.status !== 200) {
            throw new Error(`Failed to download model: ${downloadResult.status}`);
          }

          console.log('[AudioGuidance] Model downloaded successfully');
        } else {
          console.log('[AudioGuidance] Model already cached at:', modelPath);
        }

        // Load the model into Kokoro
        console.log('[AudioGuidance] Loading model into ONNX runtime...');
        const loaded = await (kokoroInstance as KokoroInstance).loadModel(`${MODEL_ID}.onnx`);

        if (!loaded) {
          throw new Error('Failed to load model into ONNX runtime');
        }

        setIsReady(true);
        setError(null);
        console.log('[AudioGuidance] Kokoro TTS ready!');
      } catch (err) {
        console.error('[AudioGuidance] Failed to initialize Kokoro:', err);
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
      console.log(`[AudioGuidance] playNextInQueue: isPlaying=${isPlayingRef.current}, queueLength=${audioQueueRef.current.length}`);
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);

    const sound = audioQueueRef.current.shift();
    if (!sound) {
      isPlayingRef.current = false;
      setIsSpeaking(false);
      return;
    }

    try {
      console.log('[AudioGuidance] Playing audio...');
      await sound.playAsync();

      sound.setOnPlaybackStatusUpdate((status: PlaybackStatus) => {
        if (status.didJustFinish) {
          console.log('[AudioGuidance] Audio playback finished');
          sound.unloadAsync();
          isPlayingRef.current = false;
          setIsSpeaking(false);
          playNextInQueue();
        }
      });
    } catch (err) {
      console.error('[AudioGuidance] Audio playback error:', err);
      isPlayingRef.current = false;
      setIsSpeaking(false);
      playNextInQueue();
    }
  };

  // Handle complete sentences
  useEffect(() => {
    console.log('[AudioGuidance] Registering sentence callback, isReady:', isReady);
    textBuffer.onSentence(async ({ sentence, emergency }) => {
      console.log(`[AudioGuidance] Sentence received: "${sentence}" (emergency: ${emergency}, isReady: ${isReady})`);
      
      if (!isReady || !kokoroInstance) {
        console.warn('[AudioGuidance] Kokoro not ready, skipping TTS. isReady:', isReady, 'kokoroInstance:', !!kokoroInstance);
        return;
      }

      try {
        console.log(`[AudioGuidance] Generating TTS for: "${sentence}"`);

        const sound = await (kokoroInstance as KokoroInstance).generateAudio(sentence, VOICE_ID, 1.0);
        console.log('[AudioGuidance] TTS audio generated successfully');

        if (emergency) {
          audioQueueRef.current.unshift(sound);
        } else {
          audioQueueRef.current.push(sound);
        }
        console.log('[AudioGuidance] Audio queued, queue length:', audioQueueRef.current.length);

        playNextInQueue();
      } catch (err) {
        console.error('[AudioGuidance] TTS generation error:', err);
      }
    });
  }, [textBuffer, isReady]);

  // Handle incoming text tokens - register callback with WebSocket
  useEffect(() => {
    if (!enabled) {
      console.log('[AudioGuidance] Disabled, not registering text token callback');
      return;
    }

    if (callbackRegisteredRef.current) {
      console.log('[AudioGuidance] Callback already registered, skipping');
      return;
    }

    console.log('[AudioGuidance] Registering text token callback with WebSocket');
    callbackRegisteredRef.current = true;
    
    onTextToken((event: TextTokenEvent) => {
      console.log(`[AudioGuidance] Received text token from WebSocket: "${event.token}"`);
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
