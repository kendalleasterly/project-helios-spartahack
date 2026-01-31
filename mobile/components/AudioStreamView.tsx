import type {
  AudioStreamViewActions,
  AudioStreamViewState,
  DeepgramStatus,
  StreamStatus,
} from "@/components/AudioStreamViewModel";
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

type AudioStreamViewProps = {
  state: AudioStreamViewState;
  actions: AudioStreamViewActions;
};

const getStatusMeta = (status: StreamStatus): StatusMeta => {
  switch (status) {
    case "recording":
      return { label: "Recording", color: "#0F766E", bg: "#CCFBF1" };
    case "stopped":
      return { label: "Stopped", color: "#0F172A", bg: "#E2E8F0" };
    case "error":
      return { label: "Error", color: "#B91C1C", bg: "#FEE2E2" };
    case "idle":
    default:
      return { label: "Idle", color: "#0F172A", bg: "#E2E8F0" };
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

export const AudioStreamView = ({ state, actions }: AudioStreamViewProps) => {
  const statusMeta = getStatusMeta(state.status);
  const deepgramStatusMeta = getDeepgramStatusMeta(state.deepgramStatus);
  const frameSampleRate = state.lastFrame ? state.lastFrame.sampleRate : null;

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.cameraScreen}>
        <Text style={styles.cameraText}>camera</Text>
      </View>
      <Modal animationType="slide" transparent visible>
        <View style={styles.modalBackdrop}>
          <View style={styles.modalSheet}>
            <ScrollView
              contentContainerStyle={styles.content}
              alwaysBounceVertical={false}
            >
              <View style={styles.header}>
                <Text style={styles.title}>Mic Stream Monitor</Text>
                <Text style={styles.subtitle}>
                  Stream status, controls, and transcription details.
                </Text>
              </View>

              <View style={styles.row}>
                <View style={[styles.pill, { backgroundColor: statusMeta.bg }]}>
                  <View
                    style={[
                      styles.statusDot,
                      { backgroundColor: statusMeta.color },
                    ]}
                  />
                  <Text style={[styles.pillText, { color: statusMeta.color }]}>
                    {statusMeta.label}
                  </Text>
                </View>
                <View
                  style={[styles.pill, { backgroundColor: deepgramStatusMeta.bg }]}
                >
                  <View
                    style={[
                      styles.statusDot,
                      { backgroundColor: deepgramStatusMeta.color },
                    ]}
                  />
                  <Text
                    style={[styles.pillText, { color: deepgramStatusMeta.color }]}
                  >
                    {deepgramStatusMeta.label}
                  </Text>
                </View>
                <View style={styles.pillMuted}>
                  <Text style={styles.pillMutedText}>
                    Permission: {state.permissionStatus}
                  </Text>
                </View>
              </View>

              <View style={styles.controls}>
                <View style={styles.buttonWrap}>
                  <Button title="Start Stream" onPress={actions.onStart} />
                </View>
                <View style={styles.buttonWrap}>
                  <Button title="Stop Stream" onPress={actions.onStop} />
                </View>
              </View>

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
    backgroundColor: "#F8FAFC",
  },
  cameraScreen: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  cameraText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#0F172A",
  },
  modalBackdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "transparent",
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
  content: {
    padding: 20,
    gap: 16,
    paddingBottom: 32,
  },
  header: {
    gap: 6,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#0F172A",
  },
  subtitle: {
    fontSize: 14,
    color: "#475569",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 10,
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
  pillMuted: {
    backgroundColor: "#E2E8F0",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  pillMutedText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#334155",
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 999,
  },
  controls: {
    flexDirection: "row",
    gap: 12,
  },
  buttonWrap: {
    flex: 1,
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
});
