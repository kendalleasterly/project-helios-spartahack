import type {
  AudioStreamViewActions,
  AudioStreamViewState,
  DeepgramStatus,
  StreamStatus,
} from "@/components/AudioStreamViewModel";
import type { ConnectionStatus as BackendStatus, DetectionUpdateEvent } from "@/hooks/useWebSocket";
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
import { useEffect, useState } from "react";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import * as Location from "expo-location";

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
  detectionData?: DetectionUpdateEvent | null;
  speedMps?: number;
};

const getMicStatusMeta = (status: StreamStatus): StatusMeta => {
	switch (status) {
		case "recording":
			return { label: "Mic Recording", color: "#0F766E", bg: "#CCFBF1" }
		case "stopped":
			return { label: "Mic Stopped", color: "#0F172A", bg: "#E2E8F0" }
		case "error":
			return { label: "Mic Error", color: "#B91C1C", bg: "#FEE2E2" }
		case "idle":
		default:
			return { label: "Mic Idle", color: "#0F172A", bg: "#E2E8F0" }
	}
}

const getDeepgramStatusMeta = (status: DeepgramStatus): StatusMeta => {
	switch (status) {
		case "open":
			return { label: "Deepgram Open", color: "#047857", bg: "#D1FAE5" }
		case "connecting":
			return { label: "Deepgram Connecting", color: "#B45309", bg: "#FEF3C7" }
		case "closed":
			return { label: "Deepgram Closed", color: "#334155", bg: "#E2E8F0" }
		case "error":
			return { label: "Deepgram Error", color: "#B91C1C", bg: "#FEE2E2" }
		case "idle":
		default:
			return { label: "Deepgram Idle", color: "#334155", bg: "#E2E8F0" }
	}
}

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

const formatHeadingValue = (value?: number | null) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${value.toFixed(1)}°`;
};

const getCardinalDirection = (heading: number) => {
  const normalized = ((heading % 360) + 360) % 360;
  const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"];
  const index = Math.round(normalized / 45);
  return directions[index];
};

const getHeadingAccuracyLabel = (accuracy?: number | null) => {
  switch (accuracy) {
    case 3:
      return "High";
    case 2:
      return "Medium";
    case 1:
      return "Low";
    case 0:
      return "Unreliable";
    default:
      return "Unknown";
  }
};

export const AudioStreamView = ({
  onFinalizeNote,
  onCancelNote,
  detectionData,
  onFrameCaptured,
  state,
  actions,
  backendStatus,
  onSendFrame,
  isDiagnosticsVisible,
  onToggleDiagnostics,
  ttsStatus,
  getPendingQuestion,
  waitingForNote,
  accumulatedNote,
  speedMps,
}: AudioStreamViewProps) => {
  const insets = useSafeAreaInsets();
  const micStatusMeta = getMicStatusMeta(state.status);
  const deepgramStatusMeta = getDeepgramStatusMeta(state.deepgramStatus);
  const backendStatusMeta = getBackendStatusMeta(backendStatus);
  const ttsStatusMeta = getTTSStatusMeta(ttsStatus);
  const frameSampleRate = state.lastFrame ? state.lastFrame.sampleRate : null;
  const diagnosticsLabel = isDiagnosticsVisible
    ? "Hide Diagnostics"
    : "Diagnostics";
  const [heading, setHeading] = useState<Location.LocationHeadingObject | null>(null);
  const [headingPermission, setHeadingPermission] = useState<
    Location.LocationPermissionResponse["status"] | "unknown"
  >("unknown");
  const [headingError, setHeadingError] = useState<string | null>(null);
  const modalContentInsetStyle = {
    paddingBottom: Math.max(insets.bottom, 32),
  };
  const overlayInsetStyle = {
    paddingTop: insets.top + 16,
    paddingBottom: insets.bottom + 16,
  };

  // Instantaneous speed from backend detection data
  const instantSpeed = detectionData?.motion?.speed_mps;

  useEffect(() => {
    let subscription: Location.LocationSubscription | null = null;
    let isActive = true;

    const startHeading = async () => {
      if (!isDiagnosticsVisible) {
        return;
      }

      try {
        setHeadingError(null);
        const permission = await Location.requestForegroundPermissionsAsync();
        if (!isActive) return;
        setHeadingPermission(permission.status);
        if (permission.status !== "granted") {
          setHeadingError("Location permission not granted");
          return;
        }

        subscription = await Location.watchHeadingAsync(
          (nextHeading) => {
            if (!isActive) return;
            setHeading(nextHeading);
          },
          (error) => {
            if (!isActive) return;
            setHeadingError(error?.message ?? "Heading error");
          },
        );
      } catch (error) {
        if (!isActive) return;
        setHeadingError(error instanceof Error ? error.message : "Failed to read heading");
      }
    };

    startHeading();

    return () => {
      isActive = false;
      subscription?.remove();
    };
  }, [isDiagnosticsVisible]);

  const trueHeadingAvailable =
    typeof heading?.trueHeading === "number" && heading.trueHeading >= 0;
  const absoluteHeading = trueHeadingAvailable ? heading?.trueHeading : heading?.magHeading;
  const absoluteHeadingLabel =
    typeof absoluteHeading === "number"
      ? `${formatHeadingValue(absoluteHeading)} ${getCardinalDirection(absoluteHeading)}`
      : "—";
  const backendHeading = detectionData?.motion?.heading_deg;
  const backendHeadingLabel =
    typeof backendHeading === "number"
      ? `${formatHeadingValue(backendHeading)} ${getCardinalDirection(backendHeading)}`
      : "—";

  return (
    <SafeAreaView style={styles.screen}>
      <View style={styles.cameraLayer}>
        <CameraView 
          onFrame={onSendFrame} 
          getPendingQuestion={getPendingQuestion}
          onFrameCaptured={onFrameCaptured}
        />

        {/* Detection Overlay */}
        <View style={StyleSheet.absoluteFill} pointerEvents="none">
          {/* Path Detection Oval (matches server logic) */}
          {/* We use a square + scaleY transform to get a true mathematical ellipse */}
          {/* matching the backend's (x^2/a^2 + y^2/b^2 <= 1) logic */}
          <View style={{
            position: 'absolute',
            left: '22%',
            width: '56%',
            aspectRatio: 1, // Start as a circle
            top: '70%',
            borderWidth: 2,
            borderColor: 'rgba(255, 255, 0, 0.4)', 
            backgroundColor: 'rgba(255, 255, 0, 0.05)',
            borderRadius: 9999, // Perfect circle
            transform: [
              { translateY: 0 }, 
              { scaleY: 2.14 } // Stretch from 56% height to 120% height (120/56 = 2.14)
            ],
          }} />
          
          {/* Excluded Bottom Region (10%) */}
          <View style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 0,
            height: '10%',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderTopWidth: 1,
            borderTopColor: 'rgba(255, 0, 0, 0.3)',
          }}>
            <Text style={{ 
              color: 'rgba(255,255,255,0.5)', 
              fontSize: 10, 
              textAlign: 'center', 
              marginTop: 4 
            }}>
              Blind Spot (Too Close)
            </Text>
          </View>

          {/* Bounding Boxes */}
          {detectionData?.objects?.map((obj, i) => {
            // YOLO coordinates are relative to 720x1280 frame
            // We need to scale them to the view percentages
            const [x1, y1, x2, y2] = obj.box;
            const left = (x1 / 720) * 100;
            const top = (y1 / 1280) * 100;
            const width = ((x2 - x1) / 720) * 100;
            const height = ((y2 - y1) / 1280) * 100;
            
            // Check hazards based on position (now 'path' vs 'peripheral')
            const isPath = obj.position === 'path';
            const isClose = obj.distance.includes('immediate') || obj.distance.includes('close');
            const isHazard = isPath && isClose;
            
            const borderColor = isHazard ? 'red' : 'rgba(0, 255, 0, 0.6)';

            return (
              <View
                key={i}
                style={{
                  position: 'absolute',
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                  borderWidth: 2,
                  borderColor: borderColor,
                  zIndex: 10,
                }}
              >
                <Text style={{ 
                  color: 'white', 
                  backgroundColor: borderColor, 
                  fontSize: 10, 
                  alignSelf: 'flex-start',
                  paddingHorizontal: 4
                }}>
                  {obj.label} ({obj.distance.split(' ')[0]})
                </Text>
              </View>
            );
          })}
        </View>
      </View>
      <View style={[styles.overlay, overlayInsetStyle]} pointerEvents="box-none">
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

          {/* Note recording status (from face-detection) */}
          {waitingForNote && (
            <View style={[styles.pill, { backgroundColor: "#FEF3C7" }]}>
              <View style={[styles.statusDot, { backgroundColor: "#D97706" }]} />
              <Text style={[styles.pillText, { color: "#D97706" }]}>
                Note for {waitingForNote}...
              </Text>
            </View>
          )}

          {/* Speed Display (from navigation) */}
          {(typeof speedMps === "number") && (
            <View style={[styles.pill, { backgroundColor: "#DBEAFE" }]}>
              <View style={[styles.statusDot, { backgroundColor: "#1D4ED8" }]} />
              <Text style={[styles.pillText, { color: "#1D4ED8" }]}>
                Speed: {speedMps.toFixed(2)} m/s
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
							<Text style={styles.modalTitle}>Diagnostics</Text>
							<Button title="Close" onPress={onToggleDiagnostics} />
						</View>
						<ScrollView
							contentContainerStyle={[
								styles.modalContent,
								modalContentInsetStyle,
							]}
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
										<Text
											key={`${line}-${index}`}
											style={styles.transcriptLine}
										>
											• {line}
										</Text>
									))
								) : (
									<Text style={styles.emptyText}>
										No final transcripts yet.
									</Text>
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

							<View style={styles.card}>
								<Text style={styles.cardTitle}>Compass</Text>
								<Text style={styles.helper}>
									Permission: {headingPermission}
								</Text>
								{headingError ? (
									<Text style={styles.errorText}>Heading error: {headingError}</Text>
								) : null}
								<Text style={styles.helper}>
									True heading:{" "}
									{trueHeadingAvailable
										? formatHeadingValue(heading?.trueHeading)
										: "Unavailable"}
								</Text>
								<Text style={styles.helper}>
									Mag heading: {formatHeadingValue(heading?.magHeading)}
								</Text>
								<Text style={styles.helper}>
									Absolute: {absoluteHeadingLabel}
								</Text>
								<Text style={styles.helper}>
									Calibration: {getHeadingAccuracyLabel(heading?.accuracy)}
								</Text>
							</View>

							<View style={styles.card}>
								<Text style={styles.cardTitle}>Heading (Backend)</Text>
								<Text style={styles.helper}>
									Heading: {backendHeadingLabel}
								</Text>
							</View>
						</ScrollView>
					</View>
				</View>
			</Modal>
		</SafeAreaView>
	)
}

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
	gridContainer: {
		...StyleSheet.absoluteFillObject,
		flexDirection: "row",
		justifyContent: "space-evenly",
	},
	gridLineVertical: {
		width: 1,
		height: "100%",
		backgroundColor: "rgba(255, 255, 255, 0.2)",
	},
	gridLineHorizontal: {
		height: 1,
		width: "100%",
		backgroundColor: "rgba(255, 255, 255, 0.2)",
	},
});
