/**
 * Audio Guidance Hook
 *
 * Orchestrates the complete TTS pipeline using native iOS/Android speech:
 * 1. Listen to text tokens from WebSocket
 * 2. Buffer tokens until sentence boundary
 * 3. Speak using native TTS
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import * as Speech from 'expo-speech';
import { VoiceQuality } from 'expo-speech';
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
  const [lastSpokenText, setLastSpokenText] = useState<string | null>(null);
  const [lastSpokenAt, setLastSpokenAt] = useState<number | null>(null);

  const textBuffer = useTextBuffer();
  const speechQueueRef = useRef<QueuedSentence[]>([]);
  const isPlayingRef = useRef(false);
  const callbackRegisteredRef = useRef(false);
  const lastStopAtRef = useRef<number | null>(null);
  const lastSpokenTextRef = useRef<string | null>(null);
  const lastSpokenAtRef = useRef<number | null>(null);
  const currentSpeechTextRef = useRef<string | null>(null);
  const currentSpeechStartedAtRef = useRef<number | null>(null);
  const REPEAT_WINDOW_MS = 5000;

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
        if (voices.length > 0) {
          const voiceSummaries = voices.map((voice) => ({
            id: voice.identifier,
            name: voice.name,
            language: voice.language,
            quality: voice.quality,
          }));
          console.log('[AudioGuidance] Voice options:', voiceSummaries);

          const enhancedVoices = voices.filter((voice) => voice.quality === VoiceQuality.Enhanced);
          console.log(`[AudioGuidance] Enhanced voices: ${enhancedVoices.length}`);
        }
        
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

    const cleanedText = nextSentence.text.replace(/^\s*guidance\s*:\s*/i, '').trim();
    if (!cleanedText) {
      playNextInQueue();
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);
    currentSpeechTextRef.current = cleanedText;
    currentSpeechStartedAtRef.current = Date.now();
    lastSpokenTextRef.current = cleanedText;
    lastSpokenAtRef.current = currentSpeechStartedAtRef.current;
    setLastSpokenText(cleanedText);
    setLastSpokenAt(currentSpeechStartedAtRef.current);

    console.log(`[AudioGuidance] Speaking: "${cleanedText}"`);

    Speech.speak(cleanedText, {
      language: 'en-US',
      rate: 1.25,
      pitch: 1.0,
      onDone: () => {
        console.log('[AudioGuidance] Speech finished');
        isPlayingRef.current = false;
        setIsSpeaking(false);
        currentSpeechTextRef.current = null;
        currentSpeechStartedAtRef.current = null;
        playNextInQueue();
      },
      onError: (err) => {
        console.error('[AudioGuidance] Speech error:', err);
        isPlayingRef.current = false;
        setIsSpeaking(false);
        currentSpeechTextRef.current = null;
        currentSpeechStartedAtRef.current = null;
        playNextInQueue();
      },
    });
  };

  const stopSpeaking = useCallback(() => {
    Speech.stop();
    isPlayingRef.current = false;
    speechQueueRef.current = [];
    textBuffer.reset();
    setIsSpeaking(false);
    currentSpeechTextRef.current = null;
    currentSpeechStartedAtRef.current = null;
  }, [textBuffer]);

  // Handle complete sentences
  useEffect(() => {
    if (!isReady) return;

    console.log('[AudioGuidance] Registering sentence callback');
    textBuffer.onSentence(({ sentence, emergency }) => {
      console.log(`[AudioGuidance] Sentence received: "${sentence}" (emergency: ${emergency})`);

      const now = Date.now();
      let cleanedSentence = sentence.trim();

      const normalized = sentence.trim().toUpperCase();
      if (normalized.startsWith('STOP')) {
        const lastStopAt = lastStopAtRef.current;
        if (lastStopAt && now - lastStopAt < 2000) {
          console.log('[AudioGuidance] Suppressing repeated STOP within 2s');
          return;
        }
        lastStopAtRef.current = now;
      }

      const stripRepeatedPrefix = (text: string, prefix: string): string | null => {
        const trimmedText = text.trim();
        const trimmedPrefix = prefix.trim();
        if (!trimmedText || !trimmedPrefix) return null;
        if (trimmedText.toLowerCase().startsWith(trimmedPrefix.toLowerCase())) {
          const remainder = trimmedText.slice(trimmedPrefix.length).replace(/^[\s,.;:!?\-–]+/, "").trim();
          return remainder;
        }
        return null;
      };

      const currentSpeech = currentSpeechTextRef.current;
      const currentSpeechAt = currentSpeechStartedAtRef.current;
      const lastSpoken = lastSpokenTextRef.current;
      const lastSpokenAtValue = lastSpokenAtRef.current;
      const recentSpeech =
        currentSpeech && currentSpeechAt && now - currentSpeechAt <= REPEAT_WINDOW_MS
          ? currentSpeech
          : lastSpoken && lastSpokenAtValue && now - lastSpokenAtValue <= REPEAT_WINDOW_MS
            ? lastSpoken
            : null;

      if (recentSpeech) {
        const remainder = stripRepeatedPrefix(cleanedSentence, recentSpeech);
        if (remainder !== null) {
          if (!remainder) {
            console.log('[AudioGuidance] Suppressing repeated sentence within 5s');
            return;
          }
          if (isPlayingRef.current) {
            speechQueueRef.current.push({ text: remainder, emergency });
            console.log('[AudioGuidance] Appending new content after repeated prefix');
            return;
          }
          cleanedSentence = remainder;
        }
      }

      const queuedSentence: QueuedSentence = { text: cleanedSentence, emergency };

      // Always interrupt current speech to avoid backlog
      Speech.stop();
      isPlayingRef.current = false;
      speechQueueRef.current = [queuedSentence];

      console.log('[AudioGuidance] Interrupting and speaking latest sentence');
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
    stopSpeaking,
    lastSpokenText,
    lastSpokenAt,
  };
}
