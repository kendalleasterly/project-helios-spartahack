import { useEffect, useState, useCallback, useRef } from "react";
import { AudioStreamView } from "@/components/AudioStreamView";
import { useAudioStreamViewModel } from "@/components/AudioStreamViewModel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAudioGuidance } from "@/hooks/useAudioGuidance";
import { useWakeWord } from "@/hooks/useWakeWord";
import { usePersonCommands } from "@/hooks/usePersonCommands";
import { useDeviceSensors } from "@/hooks/useDeviceSensors";

export function MicStreamTest() {
  const { state, actions } = useAudioStreamViewModel();
  const { socket, status: backendStatus, sendFrame, sendDeviceSensors, connect, onTextToken, detectionData } = useWebSocket();
  const [isDiagnosticsVisible, setIsDiagnosticsVisible] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  
  // Track last processed transcript content to handle array trimming correctly.
  // When the array is trimmed via slice(), indices shift but we can find our place
  // by looking for the last transcript we processed.
  const lastProcessedTranscriptRef = useRef<string | null>(null);

  // Enable sensor streaming (auto-sends to backend) from navigation
  useDeviceSensors({
    enabled: backendStatus === "connected",
    onSensorUpdate: sendDeviceSensors,
    updateInterval: 200, // 5Hz updates
  });

  // Person memory commands - monitors transcripts for "helios remember/save/note" patterns
  const { 
    processTranscript: processPersonCommand, 
    waitingForNote,
    accumulatedNote,
    finalizeNoteNow,
    cancelNoteWait,
    lastCommandResult,
  } = usePersonCommands({
    socket,
    currentFrame,
    enabled: true,
    noteAccumulationTimeout: 5000, // 5 seconds before auto-sending note
  });

  // Process final transcripts for person commands
  // Both hooks see all transcripts - patterns are mutually exclusive (both require "helios")
  useEffect(() => {
    if (state.finalTranscripts.length === 0) {
      lastProcessedTranscriptRef.current = null;
      return;
    }

    // Find where to start processing
    let startIndex = 0;
    if (lastProcessedTranscriptRef.current !== null) {
      // Find the last processed transcript in the current array
      const lastProcessedIndex = state.finalTranscripts.lastIndexOf(lastProcessedTranscriptRef.current);
      if (lastProcessedIndex !== -1) {
        // Start after the last processed transcript
        startIndex = lastProcessedIndex + 1;
      }
      // If not found, the array was likely cleared or trimmed past our marker - process all
    }

    // Process new transcripts
    for (let i = startIndex; i < state.finalTranscripts.length; i++) {
      processPersonCommand(state.finalTranscripts[i]);
    }
    
    // Remember the last transcript we processed
    if (state.finalTranscripts.length > 0) {
      lastProcessedTranscriptRef.current = state.finalTranscripts[state.finalTranscripts.length - 1];
    }
  }, [state.finalTranscripts, processPersonCommand]);

  // Wake word detection - monitors transcripts for "Helios" + general questions
  // Patterns like "helios what is this" go here (not matched by person command patterns)
  const { consumePendingQuestion } = useWakeWord({
    partialTranscript: state.partialTranscript,
    finalTranscripts: state.finalTranscripts,
  });

  // Initialize audio guidance (TTS)
  const { isSpeaking, isReady, isInitializing, error: ttsError } = useAudioGuidance({
    onTextToken,
    enabled: true,
  });

  useEffect(() => {
    connect();
  }, [connect]);

  // Log TTS status
  useEffect(() => {
    if (ttsError) {
      console.error('TTS Error:', ttsError);
    }
    if (isReady) {
      console.log('TTS ready');
    }
  }, [isReady, ttsError]);

  // Log person command results
  useEffect(() => {
    if (lastCommandResult) {
      console.log('[AudioStreamTest] Person command result:', lastCommandResult);
    }
  }, [lastCommandResult]);

  const handleToggleDiagnostics = useCallback(() => {
    setIsDiagnosticsVisible((current) => !current);
  }, []);

  const handleFrameCaptured = useCallback((base64Frame: string) => {
    setCurrentFrame(base64Frame);
  }, []);

  return (
    <AudioStreamView
      state={state}
      actions={actions}
      backendStatus={backendStatus}
      onSendFrame={sendFrame}
      isDiagnosticsVisible={isDiagnosticsVisible}
      onToggleDiagnostics={handleToggleDiagnostics}
      getPendingQuestion={consumePendingQuestion}
      onFrameCaptured={handleFrameCaptured}
      detectionData={detectionData}
      waitingForNote={waitingForNote}
      accumulatedNote={accumulatedNote}
      onFinalizeNote={finalizeNoteNow}
      onCancelNote={cancelNoteWait}
      ttsStatus={{
        isSpeaking,
        isReady,
        isInitializing,
        error: ttsError,
      }}
    />
  );
}

export const AudioStreamTest = MicStreamTest;
