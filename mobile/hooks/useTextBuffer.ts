/**
 * Text Buffer Hook for Sentence Boundary Detection
 *
 * Buffers incoming text tokens and emits complete sentences when punctuation is detected.
 * Designed for streaming TTS integration with Kokoro.
 */

import { useCallback, useRef } from 'react';

interface SentenceEvent {
  sentence: string;
  emergency: boolean;
}

export type SentenceCallback = (event: SentenceEvent) => void;

const SENTENCE_TERMINATORS = /[.!?]/;

export function useTextBuffer() {
  const bufferRef = useRef<string>('');
  const emergencyRef = useRef<boolean>(false);
  const callbackRef = useRef<SentenceCallback | null>(null);

  const addToken = useCallback((token: string, emergency: boolean) => {
    bufferRef.current += token;
    emergencyRef.current = emergencyRef.current || emergency;

    const hasSentenceTerminator = SENTENCE_TERMINATORS.test(token);

    if (hasSentenceTerminator) {
      const sentence = bufferRef.current.trim();

      if (sentence && callbackRef.current) {
        callbackRef.current({
          sentence,
          emergency: emergencyRef.current,
        });
      }

      bufferRef.current = '';
      emergencyRef.current = false;
    }
  }, []);

  const onSentence = useCallback((callback: SentenceCallback) => {
    callbackRef.current = callback;
  }, []);

  const reset = useCallback(() => {
    bufferRef.current = '';
    emergencyRef.current = false;
  }, []);

  return {
    addToken,
    onSentence,
    reset,
  };
}
