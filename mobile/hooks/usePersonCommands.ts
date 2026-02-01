/**
 * Person Commands Hook (Improved Version)
 *
 * Detects voice commands for saving people and notes, then emits Socket.IO events.
 * ALL COMMANDS REQUIRE "HELIOS" WAKE WORD.
 *
 * Features:
 * - Flexible command patterns (multiple phrasings)
 * - Note accumulation with timeout (handles pauses/stutters)
 * - Manual note finalization ("done", "send", "save")
 * - Note cancellation ("cancel", "never mind")
 *
 * Commands (all require "Helios" prefix):
 * - "Helios, remember this person as [name]" / "Helios, this is [name]"
 * - "Helios, leave a note for [name]" / "Helios, note about [name]"
 */

import { useCallback, useRef, useEffect, useState } from 'react';
import type { Socket } from 'socket.io-client';

// All patterns REQUIRE "Helios" wake word
// Order matters - more specific patterns first to avoid false positives
const SAVE_PERSON_PATTERNS = [
  // "helios, remember/save this person as [name]"
  /helios[,.]?\s+(?:remember|save)\s+this\s+person\s+as\s+(.+)/i,
  // "helios, remember/save this face as [name]"
  /helios[,.]?\s+(?:remember|save)\s+this\s+face\s+as\s+(.+)/i,
  // "helios, this is [name]"
  /helios[,.]?\s+this\s+is\s+(.+)/i,
  // "helios, remember/save [name]"
  /helios[,.]?\s+(?:remember|save)\s+(.+)/i,
];

// All note patterns REQUIRE "Helios" wake word
const LEAVE_NOTE_PATTERNS = [
  // "helios, leave/make/save/take a note for/about [name]"
  /helios[,.]?\s+(?:leave|make|save|take|add)\s+a\s+note\s+(?:for|about)\s+(.+)/i,
  // "helios, note for/about [name]"
  /helios[,.]?\s+note\s+(?:for|about)\s+(.+)/i,
];

// Words that indicate the transcript is NOT a save person command
// Helps avoid false positives like "remember to call" or "save that file"
const SAVE_PERSON_EXCLUSIONS = [
  'to ', 'that ', 'the ', 'my ', 'your ', 'this file', 'that file',
  'it ', 'them ', 'those ', 'these ', 'what ', 'when ', 'where ', 'why ', 'how ',
];

// Commands to finalize note immediately
const FINALIZE_COMMANDS = ['done', 'send', 'save', 'finish', 'that\'s it', "that's all"];

// Commands to cancel note
const CANCEL_COMMANDS = ['cancel', 'never mind', 'nevermind', 'stop', 'forget it'];

interface PersonCommandsConfig {
  socket: Socket | null;
  currentFrame: string | null;
  enabled?: boolean;
  /** Time to wait after last transcript before auto-finalizing note (default: 3000ms) */
  noteAccumulationTimeout?: number;
}

interface PersonCommandResult {
  commandDetected: boolean;
  commandType: 'save_person' | 'save_note' | null;
  personName: string | null;
  noteText: string | null;
}

interface PersonSavedResponse {
  name: string;
  person_id: string;
  message: string;
}

interface NoteSavedResponse {
  person_name: string;
  note_text: string;
  message: string;
}

interface ErrorResponse {
  message: string;
}

interface UsePersonCommandsReturn {
  /**
   * Process a transcript to check for person commands.
   * Returns true if a command was detected and handled.
   */
  processTranscript: (transcript: string) => boolean;
  /**
   * True while waiting for server response
   */
  isProcessingCommand: boolean;
  /**
   * Last command result (for UI feedback)
   */
  lastCommandResult: PersonCommandResult | null;
  /**
   * Person name we're waiting to receive a note for (two-step process)
   */
  waitingForNote: string | null;
  /**
   * Current accumulated note text (shown in UI for feedback)
   */
  accumulatedNote: string;
  /**
   * Force finalize the current note immediately
   */
  finalizeNoteNow: () => void;
  /**
   * Cancel the note wait state
   */
  cancelNoteWait: () => void;
}

/**
 * Clean up extracted name - remove punctuation, common prefixes, and validate
 */
function cleanName(raw: string): string | null {
  let name = raw.trim().replace(/[.,!?]+$/, '').trim();
  
  // Remove common prefixes like "this is", "that is", "it's", etc.
  const prefixPatterns = [
    /^this\s+is\s+/i,
    /^that\s+is\s+/i,
    /^that's\s+/i,
    /^it's\s+/i,
    /^its\s+/i,
    /^my\s+friend\s+/i,
    /^my\s+/i,
  ];
  
  for (const pattern of prefixPatterns) {
    name = name.replace(pattern, '').trim();
  }
  
  // Name should be at least 2 characters
  if (name.length < 2) {
    return null;
  }
  return name;
}

/**
 * Check if transcript contains exclusion words that indicate it's not a person command
 */
function containsExclusion(text: string): boolean {
  const lower = text.toLowerCase();
  return SAVE_PERSON_EXCLUSIONS.some((exc) => lower.includes(exc));
}

/**
 * Extract person name from a transcript matching save person patterns
 */
function extractSavePersonName(transcript: string): string | null {
  for (const pattern of SAVE_PERSON_PATTERNS) {
    const match = transcript.match(pattern);
    if (match?.[1]) {
      const name = cleanName(match[1]);
      if (name && !containsExclusion(name)) {
        return name;
      }
    }
  }
  return null;
}

/**
 * Extract person name from a transcript matching leave note patterns
 */
function extractLeaveNoteName(transcript: string): string | null {
  for (const pattern of LEAVE_NOTE_PATTERNS) {
    const match = transcript.match(pattern);
    if (match?.[1]) {
      const name = cleanName(match[1]);
      if (name) {
        return name;
      }
    }
  }
  return null;
}

/**
 * Check if transcript is a finalize command
 */
function isFinalizeCommand(transcript: string): boolean {
  const lower = transcript.toLowerCase().trim();
  return FINALIZE_COMMANDS.some((cmd) => lower === cmd || lower.startsWith(cmd));
}

/**
 * Check if transcript is a cancel command
 */
function isCancelCommand(transcript: string): boolean {
  const lower = transcript.toLowerCase().trim();
  return CANCEL_COMMANDS.some((cmd) => lower === cmd || lower.startsWith(cmd));
}

export function usePersonCommands({
  socket,
  currentFrame,
  enabled = true,
  noteAccumulationTimeout = 3000,
}: PersonCommandsConfig): UsePersonCommandsReturn {
  const [isProcessingCommand, setIsProcessingCommand] = useState(false);
  const [lastCommandResult, setLastCommandResult] = useState<PersonCommandResult | null>(null);
  const [waitingForNote, setWaitingForNote] = useState<string | null>(null);
  const [accumulatedNote, setAccumulatedNote] = useState<string>('');

  // Timeout ref for note accumulation
  const noteTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Helper to finalize note
  const finalizeNote = useCallback((personName: string, noteText: string) => {
    if (!socket || !noteText.trim()) {
      console.log('[PersonCommands] Cannot finalize: no socket or empty note');
      return;
    }

    console.log(`[PersonCommands] Finalizing note for ${personName}: "${noteText}"`);

    socket.emit('save_note', {
      person_name: personName,
      note_text: noteText.trim(),
    });

    setIsProcessingCommand(true);
    setWaitingForNote(null);
    setAccumulatedNote('');

    if (noteTimeoutRef.current) {
      clearTimeout(noteTimeoutRef.current);
      noteTimeoutRef.current = null;
    }
  }, [socket]);

  // Set up Socket.IO event listeners for responses
  useEffect(() => {
    if (!socket) return;

    const handlePersonSaved = (data: PersonSavedResponse) => {
      console.log(`[PersonCommands] Person saved: ${data.name} (${data.person_id})`);
      setIsProcessingCommand(false);
      setLastCommandResult({
        commandDetected: true,
        commandType: 'save_person',
        personName: data.name,
        noteText: null,
      });
    };

    const handleNoteSaved = (data: NoteSavedResponse) => {
      console.log(`[PersonCommands] Note saved for ${data.person_name}: "${data.note_text}"`);
      setIsProcessingCommand(false);
      setWaitingForNote(null);
      setAccumulatedNote('');
      setLastCommandResult({
        commandDetected: true,
        commandType: 'save_note',
        personName: data.person_name,
        noteText: data.note_text,
      });
    };

    const handleError = (data: ErrorResponse) => {
      console.error(`[PersonCommands] Error: ${data.message}`);
      setIsProcessingCommand(false);
      // Clear note state on error so user can retry fresh
      setWaitingForNote(null);
      setAccumulatedNote('');
    };

    socket.on('person_saved', handlePersonSaved);
    socket.on('note_saved', handleNoteSaved);
    socket.on('error', handleError);

    return () => {
      socket.off('person_saved', handlePersonSaved);
      socket.off('note_saved', handleNoteSaved);
      socket.off('error', handleError);
    };
  }, [socket]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (noteTimeoutRef.current) {
        clearTimeout(noteTimeoutRef.current);
      }
    };
  }, []);

  const cancelNoteWait = useCallback(() => {
    console.log('[PersonCommands] Cancelled note wait');
    setWaitingForNote(null);
    setAccumulatedNote('');
    if (noteTimeoutRef.current) {
      clearTimeout(noteTimeoutRef.current);
      noteTimeoutRef.current = null;
    }
  }, []);

  const finalizeNoteNow = useCallback(() => {
    if (waitingForNote && accumulatedNote) {
      finalizeNote(waitingForNote, accumulatedNote);
    }
  }, [waitingForNote, accumulatedNote, finalizeNote]);

  const processTranscript = useCallback((transcript: string): boolean => {
    if (!enabled || !socket) {
      return false;
    }

    const normalizedTranscript = transcript.toLowerCase().trim();

    // Skip empty transcripts
    if (!normalizedTranscript) {
      return false;
    }

    // If we're waiting for note content, handle done/cancel commands FIRST
    // These should ALWAYS work, even if seen before (no deduplication)
    if (waitingForNote) {
      // Check for manual finalization commands
      if (isFinalizeCommand(transcript)) {
        console.log('[PersonCommands] Manual finalization triggered via "done"/"send"');
        if (accumulatedNote) {
          finalizeNote(waitingForNote, accumulatedNote);
        } else {
          console.log('[PersonCommands] Nothing to finalize, no note accumulated yet');
        }
        return true;
      }

      // Check for cancellation commands
      if (isCancelCommand(transcript)) {
        console.log('[PersonCommands] Note cancelled by user via "cancel"/"never mind"');
        cancelNoteWait();
        return true;
      }
    }

    // Note: Deduplication is handled by the parent component using index-based tracking
    // This allows the same command to be repeated (e.g., "helios remember john" twice)

    // If we're waiting for note content, accumulate it
    if (waitingForNote) {
      // Accumulate note content
      const newNote = accumulatedNote
        ? `${accumulatedNote} ${transcript}`
        : transcript;

      setAccumulatedNote(newNote);
      console.log(`[PersonCommands] Accumulating note: "${newNote}"`);

      // Clear existing timeout
      if (noteTimeoutRef.current) {
        clearTimeout(noteTimeoutRef.current);
      }

      // Set new timeout for auto-finalization
      // Capture current waitingForNote value for the timeout callback
      const currentPersonName = waitingForNote;
      noteTimeoutRef.current = setTimeout(() => {
        console.log(`[PersonCommands] Timeout reached, finalizing note for ${currentPersonName}`);
        finalizeNote(currentPersonName, newNote);
      }, noteAccumulationTimeout);

      return true;
    }

    // Check for "save person" command
    const savePersonName = extractSavePersonName(transcript);
    if (savePersonName) {
      if (!currentFrame) {
        console.warn('[PersonCommands] No frame available to save person');
        return false;
      }

      console.log(`[PersonCommands] Saving person: ${savePersonName}`);
      setIsProcessingCommand(true);
      socket.emit('save_person', {
        name: savePersonName,
        frame: currentFrame,
      });
      return true;
    }

    // Check for "leave note" command
    const leaveNoteName = extractLeaveNoteName(transcript);
    if (leaveNoteName) {
      console.log(`[PersonCommands] Ready to save note for: ${leaveNoteName}`);
      console.log(`[PersonCommands] Listening for note content (${noteAccumulationTimeout}ms timeout)...`);
      setWaitingForNote(leaveNoteName);
      setAccumulatedNote('');
      return true;
    }

    return false;
  }, [enabled, socket, currentFrame, waitingForNote, accumulatedNote, noteAccumulationTimeout, finalizeNote, cancelNoteWait]);

  return {
    processTranscript,
    isProcessingCommand,
    lastCommandResult,
    waitingForNote,
    accumulatedNote,
    finalizeNoteNow,
    cancelNoteWait,
  };
}
