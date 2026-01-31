import { AudioStreamView } from "@/components/AudioStreamView";
import { useAudioStreamViewModel } from "@/components/AudioStreamViewModel";

export function MicStreamTest() {
  const { state, actions } = useAudioStreamViewModel();
  return <AudioStreamView state={state} actions={actions} />;
}

export const AudioStreamTest = MicStreamTest;
