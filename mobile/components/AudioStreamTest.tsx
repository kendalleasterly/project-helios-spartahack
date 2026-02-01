import { useEffect, useState, useCallback, useRef } from "react";
import { AudioStreamView } from "@/components/AudioStreamView";
import { useAudioStreamViewModel } from "@/components/AudioStreamViewModel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAudioGuidance } from "@/hooks/useAudioGuidance";
import { usePersonCommands } from "@/hooks/usePersonCommands";
import { useDeviceSensors } from "@/hooks/useDeviceSensors";
import { useWakeWord } from "@/hooks/useWakeWord";
import * as Haptics from "expo-haptics";

export function MicStreamTest() {
  const { state, actions } = useAudioStreamViewModel();
  const { socket, status: backendStatus, sendFrame, sendDeviceSensors, connect, onTextToken, onHaptic, detectionData } = useWebSocket();
  const [isDiagnosticsVisible, setIsDiagnosticsVisible] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  
  // Track last processed transcript content to handle array trimming correctly.
  // When the array is trimmed via slice(), indices shift but we can find our place
  // by looking for the last transcript we processed.
  const lastProcessedTranscriptRef = useRef<string | null>(null);

  // Enable sensor streaming (auto-sends to backend)
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
  // Person commands still require "helios".
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
      const transcript = state.finalTranscripts[i];
      processPersonCommand(transcript);
    }
    
    // Remember the last transcript we processed
    if (state.finalTranscripts.length > 0) {
      lastProcessedTranscriptRef.current = state.finalTranscripts[state.finalTranscripts.length - 1];
    }
  }, [state.finalTranscripts, processPersonCommand]);

  const runBurst = useCallback(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    setTimeout(() => {
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    }, 120);
    setTimeout(() => {
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    }, 240);
  }, []);

  // Initialize audio guidance (TTS)
  const { isSpeaking, isReady, isInitializing, error: ttsError, stopSpeaking } = useAudioGuidance({
    onTextToken,
    enabled: true,
  });

  // Wake word detection - monitors transcripts for "Helios" + general questions
  const { consumePendingQuestion } = useWakeWord({
    partialTranscript: state.partialTranscript,
    finalTranscripts: state.finalTranscripts,
    onWakeWord: stopSpeaking,
  });

  useEffect(() => {
    connect();
  }, [connect]);

  useEffect(() => {
    onHaptic((event) => {
      if (!event?.pattern || event.pattern === "burst") {
        runBurst();
        return;
      }
      if (event.pattern === "long-burst") {
        runBurst();
        setTimeout(() => runBurst(), 380);
        return;
      }
      if (event.pattern === "heavy") {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
      } else if (event.pattern === "warning") {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      } else if (event.pattern === "error") {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      } else {
        runBurst();
      }
    });
  }, [onHaptic, runBurst]);

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
