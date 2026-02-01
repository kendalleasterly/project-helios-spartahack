/**
 * Wake Word Detection Hook
 *
 * Monitors transcripts for "Helios" wake word and extracts the question.
 * When detected, provides the question to be sent with the next frame.
 */

import { useCallback, useRef, useEffect } from 'react';

const WAKE_WORD = 'helios';
const WAKE_TIMEOUT_MS = 4000;
const WAKE_INTERRUPT_DEBOUNCE_MS = 800;

interface UseWakeWordOptions {
  partialTranscript: string;
  finalTranscripts: string[];
  onWakeWord?: () => void;
}

interface UseWakeWordReturn {
  /**
   * Get the pending question if one was detected.
   * Returns undefined if no question pending.
   * Calling this clears the pending question (consume once).
   */
  consumePendingQuestion: () => string | undefined;
}

/**
 * Extracts question text after the wake word "Helios"
 * Returns the question or undefined if no wake word found
 */
function extractQuestionFromText(text: string): { question?: string; wakeOnly?: boolean } {
  const lowerText = text.toLowerCase();
  const wakeWordIndex = lowerText.indexOf(WAKE_WORD);

  if (wakeWordIndex === -1) {
    return {};
  }

  // Extract everything after "helios"
  const afterWakeWord = text.slice(wakeWordIndex + WAKE_WORD.length).trim();

  // Remove leading punctuation/comma if present
  const cleaned = afterWakeWord.replace(/^[,.\s]+/, '').trim();

  // Only return if there's actual content
  if (cleaned.length > 0) {
    return { question: cleaned };
  }

  return { wakeOnly: true };
}

export function useWakeWord({ partialTranscript, finalTranscripts, onWakeWord }: UseWakeWordOptions): UseWakeWordReturn {
  const pendingQuestionRef = useRef<string | undefined>(undefined);
  const pendingWakeRef = useRef<boolean>(false);
  const pendingWakeTsRef = useRef<number | null>(null);
  const processedTranscriptsRef = useRef<Set<string>>(new Set());
  const lastInterruptAtRef = useRef<number | null>(null);

  const triggerWakeInterrupt = useCallback(() => {
    const now = Date.now();
    const last = lastInterruptAtRef.current;
    if (last && now - last < WAKE_INTERRUPT_DEBOUNCE_MS) {
      return;
    }
    lastInterruptAtRef.current = now;
    if (typeof onWakeWord === "function") {
      onWakeWord();
    }
  }, [onWakeWord]);

  // Check final transcripts for wake word
  useEffect(() => {
    console.log(`[WakeWord] finalTranscripts changed, count: ${finalTranscripts.length}`);
    for (const transcript of finalTranscripts) {
      // Skip already processed transcripts
      if (processedTranscriptsRef.current.has(transcript)) {
        console.log(`[WakeWord] Skipping already processed: "${transcript}"`);
        continue;
      }
      console.log(`[WakeWord] Processing new transcript: "${transcript}"`);
      processedTranscriptsRef.current.add(transcript);

      const now = Date.now();
      if (pendingWakeRef.current) {
        const wakeAge = pendingWakeTsRef.current ? now - pendingWakeTsRef.current : Infinity;
        pendingWakeRef.current = false;
        pendingWakeTsRef.current = null;
        if (wakeAge <= WAKE_TIMEOUT_MS) {
          const followupResult = extractQuestionFromText(transcript);
          if (followupResult.question) {
            console.log(`[WakeWord] Using follow-up with wake word: "${followupResult.question}"`);
            pendingQuestionRef.current = followupResult.question;
          } else {
            console.log(`[WakeWord] Using transcript after wake word: "${transcript}"`);
            pendingQuestionRef.current = transcript;
          }
          continue;
        }
      }

      const result = extractQuestionFromText(transcript);
      if (result.question) {
        triggerWakeInterrupt();
        console.log(`[WakeWord] Detected question: "${result.question}"`);
        pendingQuestionRef.current = result.question;
      } else if (result.wakeOnly) {
        triggerWakeInterrupt();
        console.log(`[WakeWord] Wake word only, waiting for next transcript`);
        pendingWakeRef.current = true;
        pendingWakeTsRef.current = now;
      } else {
        console.log(`[WakeWord] No wake word in transcript: "${transcript}"`);
      }
    }
  }, [finalTranscripts, triggerWakeInterrupt]);

  // Also check partial transcript for early detection
  useEffect(() => {
    if (!partialTranscript) return;

    const result = extractQuestionFromText(partialTranscript);
    if (result.question && result.question.length > 3) {
      // Only log for debugging, don't set pending yet (wait for final)
      triggerWakeInterrupt();
      console.log(`[WakeWord] Partial detection: "${result.question}"`);
    } else if (result.wakeOnly) {
      if (!pendingWakeRef.current) {
        pendingWakeRef.current = true;
        pendingWakeTsRef.current = Date.now();
        triggerWakeInterrupt();
        console.log("[WakeWord] Partial wake word detected, waiting for next transcript");
      }
    }
  }, [partialTranscript, triggerWakeInterrupt]);

  const consumePendingQuestion = useCallback((): string | undefined => {
    const question = pendingQuestionRef.current;
    if (question) {
      console.log(`[WakeWord] Consuming question: "${question}"`);
      pendingQuestionRef.current = undefined;
    }
    return question;
  }, []);

  return {
    consumePendingQuestion,
  };
}
