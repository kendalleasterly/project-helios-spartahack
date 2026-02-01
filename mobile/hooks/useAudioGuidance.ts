/**
 * Audio Guidance Hook
 *
 * Orchestrates the complete TTS pipeline using native iOS/Android speech:
 * 1. Listen to text tokens from WebSocket
 * 2. Buffer tokens until sentence boundary
 * 3. Speak using native TTS
 */

import { useEffect, useRef, useState } from 'react';
import * as Speech from 'expo-speech';
import { Audio } from 'expo-av';
import { useTextBuffer } from './useTextBuffer';
import type { TextTokenEvent } from './useWebSocket';

interface UseAudioGuidanceOptions {
  onTextToken: (callback: (event: TextTokenEvent) => void) => void;
  enabled?: boolean;
}

interface QueuedSentence {
  text: string;
  emergency: boolean;
}

let audioModeConfigured = false;

export function useAudioGuidance({ onTextToken, enabled = true }: UseAudioGuidanceOptions) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const textBuffer = useTextBuffer();
  const speechQueueRef = useRef<QueuedSentence[]>([]);
  const isPlayingRef = useRef(false);
  const callbackRegisteredRef = useRef(false);
  const lastStopAtRef = useRef<number | null>(null);

  // Initialize - configure audio mode and check if speech is available
  useEffect(() => {
    if (!enabled) return;

    async function initSpeech() {
      try {
        // Configure audio session to bypass ringer/silent mode
        if (!audioModeConfigured) {
          console.log('[AudioGuidance] Configuring audio session to bypass ringer...');
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

        console.log('[AudioGuidance] Checking native TTS availability...');
        const voices = await Speech.getAvailableVoicesAsync();
        console.log(`[AudioGuidance] Found ${voices.length} voices`);
        
        // Find a good English voice
        const englishVoices = voices.filter(v => v.language.startsWith('en'));
        console.log(`[AudioGuidance] Found ${englishVoices.length} English voices`);
        
        setIsReady(true);
        setError(null);
        console.log('[AudioGuidance] Native TTS ready');
      } catch (err) {
        console.error('[AudioGuidance] Failed to initialize native TTS:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsReady(false);
      }
    }

    initSpeech();
  }, [enabled]);

  // Play next sentence in queue
  const playNextInQueue = () => {
    if (isPlayingRef.current || speechQueueRef.current.length === 0) {
      return;
    }

    const nextSentence = speechQueueRef.current.shift();
    if (!nextSentence) {
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);

    console.log(`[AudioGuidance] Speaking: "${nextSentence.text}"`);

    Speech.speak(nextSentence.text, {
      language: 'en-US',
      rate: nextSentence.emergency ? 1.2 : 1.0,
      pitch: 1.0,
      onDone: () => {
        console.log('[AudioGuidance] Speech finished');
        isPlayingRef.current = false;
        setIsSpeaking(false);
        playNextInQueue();
      },
      onError: (err) => {
        console.error('[AudioGuidance] Speech error:', err);
        isPlayingRef.current = false;
        setIsSpeaking(false);
        playNextInQueue();
      },
    });
  };

  // Handle complete sentences
  useEffect(() => {
    if (!isReady) return;

    console.log('[AudioGuidance] Registering sentence callback');
    textBuffer.onSentence(({ sentence, emergency }) => {
      console.log(`[AudioGuidance] Sentence received: "${sentence}" (emergency: ${emergency})`);

      const normalized = sentence.trim().replace(/[.!?]+$/, '').toUpperCase();
      if (normalized === 'STOP') {
        const now = Date.now();
        const lastStopAt = lastStopAtRef.current;
        if (lastStopAt && now - lastStopAt < 3000) {
          console.log('[AudioGuidance] Suppressing repeated STOP within 3s');
          return;
        }
        lastStopAtRef.current = now;
      }

      const queuedSentence: QueuedSentence = { text: sentence, emergency };

      if (emergency) {
        // Emergency: stop current speech and jump to front
        Speech.stop();
        isPlayingRef.current = false;
        speechQueueRef.current.unshift(queuedSentence);
      } else {
        speechQueueRef.current.push(queuedSentence);
      }

      console.log(`[AudioGuidance] Queue length: ${speechQueueRef.current.length}`);
      playNextInQueue();
    });
  }, [textBuffer, isReady]);

  // Handle incoming text tokens - register callback with WebSocket
  useEffect(() => {
    if (!enabled) {
      console.log('[AudioGuidance] Disabled, not registering text token callback');
      return;
    }

    if (callbackRegisteredRef.current) {
      return;
    }

    console.log('[AudioGuidance] Registering text token callback with WebSocket');
    callbackRegisteredRef.current = true;

    onTextToken((event: TextTokenEvent) => {
      console.log(`[AudioGuidance] Received text token: "${event.token}"`);
      textBuffer.addToken(event.token, event.emergency);
    });
  }, [onTextToken, textBuffer, enabled]);

  return {
    isSpeaking,
    isReady,
    isInitializing: false, // No initialization needed for native TTS
    error,
  };
}
