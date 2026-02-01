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
  // Track the last processed transcript content to handle array trimming correctly.
  // When the array is trimmed via slice(), indices shift but we can find our place
  // by looking for the last transcript we processed.
  const lastProcessedTranscriptRef = useRef<string | null>(null);

  // Check final transcripts for wake word
  useEffect(() => {
    if (finalTranscripts.length === 0) {
      lastProcessedTranscriptRef.current = null;
      return;
    }

    // Find where to start processing
    let startIndex = 0;
    if (lastProcessedTranscriptRef.current !== null) {
      // Find the last processed transcript in the current array
      const lastProcessedIndex = finalTranscripts.lastIndexOf(lastProcessedTranscriptRef.current);
      if (lastProcessedIndex !== -1) {
        // Start after the last processed transcript
        startIndex = lastProcessedIndex + 1;
      }
      // If not found, the array was likely cleared or trimmed past our marker - process all
    }

    // Process new transcripts
    for (let i = startIndex; i < finalTranscripts.length; i++) {
      const transcript = finalTranscripts[i];
      
      const hasWakeWord = transcript.toLowerCase().includes(WAKE_WORD);
      if (hasWakeWord) {
        console.log(`[WakeWord] Found "helios" in transcript: "${transcript}"`);
        
        const question = extractQuestionFromText(transcript);
        if (question) {
          console.log(`[WakeWord] Extracted question: "${question}"`);
          pendingQuestionRef.current = question;
        }
      }
    }
    
    // Remember the last transcript we processed
    if (finalTranscripts.length > 0) {
      lastProcessedTranscriptRef.current = finalTranscripts[finalTranscripts.length - 1];
    }
  }, [finalTranscripts]);

  // Also check partial transcript for early detection (silent, no logging)
  useEffect(() => {
    if (!partialTranscript) return;
    // Early detection is just for potential future use, no action needed
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
