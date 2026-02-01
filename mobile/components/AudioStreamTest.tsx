import { useEffect, useState } from "react";
import { AudioStreamView } from "@/components/AudioStreamView";
import { useAudioStreamViewModel } from "@/components/AudioStreamViewModel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAudioGuidance } from "@/hooks/useAudioGuidance";
import { useWakeWord } from "@/hooks/useWakeWord";

export function MicStreamTest() {
  const { state, actions } = useAudioStreamViewModel();
  const { status: backendStatus, sendFrame, connect, onTextToken } = useWebSocket();
  const [isDiagnosticsVisible, setIsDiagnosticsVisible] = useState(false);

  // Wake word detection - monitors transcripts for "Helios"
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
      console.log('✓ Kokoro TTS ready');
    }
  }, [isReady, ttsError]);

  const handleToggleDiagnostics = () => {
    setIsDiagnosticsVisible((current) => !current);
  };

  return (
    <AudioStreamView
      state={state}
      actions={actions}
      backendStatus={backendStatus}
      onSendFrame={sendFrame}
      isDiagnosticsVisible={isDiagnosticsVisible}
      onToggleDiagnostics={handleToggleDiagnostics}
      getPendingQuestion={consumePendingQuestion}
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
