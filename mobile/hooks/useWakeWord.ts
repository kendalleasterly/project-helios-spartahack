/**
 * Wake Word Detection Hook
 *
 * Monitors transcripts for "Helios" wake word and extracts the question.
 * When detected, provides the question to be sent with the next frame.
 */

import { useCallback, useRef, useEffect } from 'react';

const WAKE_WORD = 'helios';

interface UseWakeWordOptions {
  partialTranscript: string;
  finalTranscripts: string[];
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
function extractQuestionFromText(text: string): string | undefined {
  const lowerText = text.toLowerCase();
  const wakeWordIndex = lowerText.indexOf(WAKE_WORD);

  if (wakeWordIndex === -1) {
    return undefined;
  }

  // Extract everything after "helios"
  const afterWakeWord = text.slice(wakeWordIndex + WAKE_WORD.length).trim();

  // Remove leading punctuation/comma if present
  const cleaned = afterWakeWord.replace(/^[,.\s]+/, '').trim();

  // Only return if there's actual content
  if (cleaned.length > 0) {
    return cleaned;
  }

  return undefined;
}

export function useWakeWord({ partialTranscript, finalTranscripts }: UseWakeWordOptions): UseWakeWordReturn {
  const pendingQuestionRef = useRef<string | undefined>(undefined);
  const processedTranscriptsRef = useRef<Set<string>>(new Set());

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

      const question = extractQuestionFromText(transcript);
      if (question) {
        console.log(`[WakeWord] Detected question: "${question}"`);
        pendingQuestionRef.current = question;
      } else {
        console.log(`[WakeWord] No wake word in transcript: "${transcript}"`);
      }
    }
  }, [finalTranscripts]);

  // Also check partial transcript for early detection
  useEffect(() => {
    if (!partialTranscript) return;

    const question = extractQuestionFromText(partialTranscript);
    if (question && question.length > 3) {
      // Only log for debugging, don't set pending yet (wait for final)
      console.log(`[WakeWord] Partial detection: "${question}"`);
    }
  }, [partialTranscript]);

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
