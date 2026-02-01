import type {
  AudioStreamViewActions,
  AudioStreamViewState,
  DeepgramStatus,
  StreamStatus,
} from "@/components/AudioStreamViewModel";
import type { ConnectionStatus as BackendStatus } from "@/hooks/useWebSocket";
import CameraView from "@/components/CameraView";
import {
  Button,
  Modal,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

type StatusMeta = {
  label: string;
  color: string;
  bg: string;
};

type TTSStatus = {
  isSpeaking: boolean;
  isReady: boolean;
  isInitializing: boolean;
  error: string | null;
};

type AudioStreamViewProps = {
  state: AudioStreamViewState;
  actions: AudioStreamViewActions;
  backendStatus: BackendStatus;
  onSendFrame: (base64Frame: string, userQuestion?: string, debug?: boolean) => void;
  isDiagnosticsVisible: boolean;
  onToggleDiagnostics: () => void;
  ttsStatus?: TTSStatus;
  getPendingQuestion?: () => string | undefined;
  onFrameCaptured?: (base64Frame: string) => void;
  waitingForNote?: string | null;
  accumulatedNote?: string;
  onFinalizeNote?: () => void;
  onCancelNote?: () => void;
};

const getMicStatusMeta = (status: StreamStatus): StatusMeta => {
  switch (status) {
    case "recording":
      return { label: "Mic Recording", color: "#0F766E", bg: "#CCFBF1" };
    case "stopped":
      return { label: "Mic Stopped", color: "#0F172A", bg: "#E2E8F0" };
    case "error":
      return { label: "Mic Error", color: "#B91C1C", bg: "#FEE2E2" };
    case "idle":
    default:
      return { label: "Mic Idle", color: "#0F172A", bg: "#E2E8F0" };
  }
};

const getDeepgramStatusMeta = (status: DeepgramStatus): StatusMeta => {
  switch (status) {
    case "open":
      return { label: "Deepgram Open", color: "#047857", bg: "#D1FAE5" };
    case "connecting":
      return { label: "Deepgram Connecting", color: "#B45309", bg: "#FEF3C7" };
    case "closed":
      return { label: "Deepgram Closed", color: "#334155", bg: "#E2E8F0" };
    case "error":
      return { label: "Deepgram Error", color: "#B91C1C", bg: "#FEE2E2" };
    case "idle":
    default:
      return { label: "Deepgram Idle", color: "#334155", bg: "#E2E8F0" };
  }
};

const getBackendStatusMeta = (status: BackendStatus): StatusMeta => {
  switch (status) {
    case "connected":
      return { label: "Backend Connected", color: "#0F766E", bg: "#CCFBF1" };
    case "connecting":
      return { label: "Backend Connecting", color: "#B45309", bg: "#FEF3C7" };
    case "error":
      return { label: "Backend Error", color: "#B91C1C", bg: "#FEE2E2" };
    case "disconnected":
    default:
      return { label: "Backend Disconnected", color: "#334155", bg: "#E2E8F0" };
  }
};

const getTTSStatusMeta = (tts?: TTSStatus): StatusMeta => {
  if (!tts) return { label: "TTS Disabled", color: "#64748B", bg: "#F1F5F9" };
  if (tts.error) return { label: "TTS Error", color: "#B91C1C", bg: "#FEE2E2" };
  if (tts.isInitializing) return { label: "TTS Loading", color: "#B45309", bg: "#FEF3C7" };
  if (tts.isSpeaking) return { label: "Speaking", color: "#047857", bg: "#D1FAE5" };
  if (tts.isReady) return { label: "TTS Ready", color: "#0F766E", bg: "#CCFBF1" };
  return { label: "TTS Idle", color: "#64748B", bg: "#F1F5F9" };
};

export const AudioStreamView = ({
  state,
  actions,
  backendStatus,
  onSendFrame,
  isDiagnosticsVisible,
  onToggleDiagnostics,
  ttsStatus,
  getPendingQuestion,
  onFrameCaptured,
  waitingForNote,
  accumulatedNote,
  onFinalizeNote,
  onCancelNote,
}: AudioStreamViewProps) => {
  const micStatusMeta = getMicStatusMeta(state.status);
  const deepgramStatusMeta = getDeepgramStatusMeta(state.deepgramStatus);
  const backendStatusMeta = getBackendStatusMeta(backendStatus);
  const ttsStatusMeta = getTTSStatusMeta(ttsStatus);
  const frameSampleRate = state.lastFrame ? state.lastFrame.sampleRate : null;
  const diagnosticsLabel = isDiagnosticsVisible
    ? "Hide Diagnostics"
    : "Audio Diagnostics";

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.cameraLayer}>
        <CameraView 
          onFrame={onSendFrame} 
          getPendingQuestion={getPendingQuestion}
          onFrameCaptured={onFrameCaptured}
        />
      </View>
      <View style={styles.overlay} pointerEvents="box-none">
        <View style={styles.statusRow}>
          <View style={[styles.pill, { backgroundColor: micStatusMeta.bg }]}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: micStatusMeta.color },
              ]}
            />
            <Text style={[styles.pillText, { color: micStatusMeta.color }]}>
              {micStatusMeta.label}
            </Text>
          </View>
          <View style={[styles.pill, { backgroundColor: backendStatusMeta.bg }]}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: backendStatusMeta.color },
              ]}
            />
            <Text style={[styles.pillText, { color: backendStatusMeta.color }]}>
              {backendStatusMeta.label}
            </Text>
          </View>
          <View style={[styles.pill, { backgroundColor: ttsStatusMeta.bg }]}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: ttsStatusMeta.color },
              ]}
            />
            <Text style={[styles.pillText, { color: ttsStatusMeta.color }]}>
              {ttsStatusMeta.label}
            </Text>
          </View>
          <View style={[styles.pill, { backgroundColor: deepgramStatusMeta.bg }]}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: deepgramStatusMeta.color },
              ]}
            />
            <Text style={[styles.pillText, { color: deepgramStatusMeta.color }]}>
              {deepgramStatusMeta.label}
            </Text>
          </View>
          {waitingForNote && (
            <View style={[styles.pill, { backgroundColor: "#FEF3C7" }]}>
              <View style={[styles.statusDot, { backgroundColor: "#D97706" }]} />
              <Text style={[styles.pillText, { color: "#D97706" }]}>
                Note for {waitingForNote}...
              </Text>
            </View>
          )}
        </View>

        {/* Note accumulation feedback panel */}
        {waitingForNote && (
          <View style={styles.notePanel}>
            <Text style={styles.notePanelTitle}>
              Recording note for {waitingForNote}
            </Text>
            <Text style={styles.notePanelText}>
              {accumulatedNote || "(listening...)"}
            </Text>
            <View style={styles.notePanelButtons}>
              {onFinalizeNote && accumulatedNote && (
                <View style={styles.noteButtonWrap}>
                  <Button title="Done" onPress={onFinalizeNote} />
                </View>
              )}
              {onCancelNote && (
                <View style={styles.noteButtonWrap}>
                  <Button title="Cancel" onPress={onCancelNote} color="#B91C1C" />
                </View>
              )}
            </View>
          </View>
        )}

        <View style={styles.controlPanel}>
          <View style={styles.controls}>
            <View style={styles.buttonWrap}>
              <Button title="Start Stream" onPress={actions.onStart} />
            </View>
            <View style={styles.buttonWrap}>
              <Button title="Stop Stream" onPress={actions.onStop} />
            </View>
          </View>
          <View style={styles.buttonRow}>
            <Button title={diagnosticsLabel} onPress={onToggleDiagnostics} />
          </View>
        </View>
      </View>

      <Modal
        animationType="slide"
        transparent
        visible={isDiagnosticsVisible}
        onRequestClose={onToggleDiagnostics}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalSheet}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Audio Diagnostics</Text>
              <Button title="Close" onPress={onToggleDiagnostics} />
            </View>
            <ScrollView
              contentContainerStyle={styles.modalContent}
              alwaysBounceVertical={false}
            >
              <View style={styles.card}>
                <Text style={styles.cardTitle}>Transcription</Text>
                <Text style={styles.label}>Partial</Text>
                <Text style={styles.transcriptText}>
                  {state.partialTranscript
                    ? state.partialTranscript
                    : "Listening for speech..."}
                </Text>
                <Text style={styles.label}>Final</Text>
                {state.finalTranscripts.length > 0 ? (
                  state.finalTranscripts.map((line, index) => (
                    <Text key={`${line}-${index}`} style={styles.transcriptLine}>
                      • {line}
                    </Text>
                  ))
                ) : (
                  <Text style={styles.emptyText}>No final transcripts yet.</Text>
                )}
                {state.deepgramError ? (
                  <Text style={styles.errorText}>{state.deepgramError}</Text>
                ) : (
                  <Text style={styles.helper}>
                    Model: nova-3 · Language: English · Interim results enabled
                  </Text>
                )}
              </View>

              <View style={styles.card}>
                <Text style={styles.cardTitle}>Diagnostics</Text>
                <Text style={styles.helper}>
                  Permission: {state.permissionStatus}
                </Text>
                <Text style={styles.helper}>
                  Listener status:{" "}
                  {state.status === "recording" ? "active" : "inactive"}
                </Text>
                <Text style={styles.helper}>
                  Mic error: {state.micError ?? "None"}
                </Text>
                <Text style={styles.helper}>
                  Deepgram status: {state.deepgramStatus}
                </Text>
                <Text style={styles.helper}>
                  Deepgram sample rate:{" "}
                  {state.deepgramSampleRate
                    ? `${state.deepgramSampleRate} Hz`
                    : "Unknown"}
                </Text>
                <Text style={styles.helper}>
                  Frame sample rate:{" "}
                  {frameSampleRate ? `${frameSampleRate} Hz` : "Unknown"}
                </Text>
                <Text style={styles.helper}>
                  Deepgram auth: {state.deepgramAuthMode}
                </Text>
                <Text style={styles.helper}>
                  Deepgram error: {state.deepgramError ?? "None"}
                </Text>
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#000000",
  },
  cameraLayer: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    padding: 16,
    justifyContent: "space-between",
  },
  statusRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 8,
  },
  pill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  pillText: {
    fontSize: 12,
    fontWeight: "700",
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
  },
  controlPanel: {
    backgroundColor: "rgba(15, 23, 42, 0.7)",
    borderRadius: 16,
    padding: 12,
    gap: 10,
  },
  controls: {
    flexDirection: "row",
    gap: 10,
  },
  buttonWrap: {
    flex: 1,
  },
  buttonRow: {
    alignSelf: "stretch",
  },
  modalBackdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(15, 23, 42, 0.35)",
  },
  modalSheet: {
    maxHeight: "85%",
    backgroundColor: "#F8FAFC",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderWidth: 1,
    borderColor: "#E2E8F0",
    overflow: "hidden",
  },
  modalHeader: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#E2E8F0",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#0F172A",
  },
  modalContent: {
    padding: 16,
    gap: 16,
    paddingBottom: 32,
  },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 16,
    gap: 12,
    shadowColor: "#0F172A",
    shadowOpacity: 0.08,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 3,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#0F172A",
  },
  label: {
    fontSize: 12,
    color: "#64748B",
  },
  transcriptText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#0F172A",
  },
  transcriptLine: {
    fontSize: 14,
    color: "#0F172A",
  },
  emptyText: {
    fontSize: 14,
    color: "#94A3B8",
  },
  errorText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#B91C1C",
  },
  helper: {
    fontSize: 12,
    color: "#64748B",
  },
  notePanel: {
    backgroundColor: "rgba(251, 191, 36, 0.9)",
    borderRadius: 16,
    padding: 16,
    gap: 8,
  },
  notePanelTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#78350F",
  },
  notePanelText: {
    fontSize: 16,
    color: "#451A03",
    fontStyle: "italic",
  },
  notePanelButtons: {
    flexDirection: "row",
    gap: 10,
    marginTop: 8,
  },
  noteButtonWrap: {
    flex: 1,
  },
});
