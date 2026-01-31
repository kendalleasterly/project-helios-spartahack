import { useEffect, useState } from "react";
import { AudioStreamView } from "@/components/AudioStreamView";
import { useAudioStreamViewModel } from "@/components/AudioStreamViewModel";
import { useWebSocket } from "@/hooks/useWebSocket";

export function MicStreamTest() {
  const { state, actions } = useAudioStreamViewModel();
  const { status: backendStatus, sendFrame, connect } = useWebSocket();
  const [isDiagnosticsVisible, setIsDiagnosticsVisible] = useState(false);

  useEffect(() => {
    connect();
  }, [connect]);

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
    />
  );
}

export const AudioStreamTest = MicStreamTest;
