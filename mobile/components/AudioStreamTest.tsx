import {
  addErrorListener,
  addFrameListener,
  getStatus,
  requestPermission,
  start,
  stop,
  type AudioFrameEvent,
  type StreamStatus,
} from "expo-stream-audio";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  PermissionsAndroid,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

type AudioFrameSummary = {
  sampleRate: number;
  length: number;
  timestamp: number;
  level: number | null;
};

type AudioStreamTestViewModel = {
  status: StreamStatus;
  isStarting: boolean;
  frameCount: number;
  lastFrame: AudioFrameSummary | null;
  logMessage: string | null;
  onStart: () => void;
  onStop: () => void;
};

type AudioStreamTestViewProps = {
  status: StreamStatus;
  isStarting: boolean;
  frameCount: number;
  lastFrame: AudioFrameSummary | null;
  logMessage: string | null;
  onStart: () => void;
  onStop: () => void;
};

const useAudioStreamTestViewModel = (): AudioStreamTestViewModel => {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [isStarting, setIsStarting] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [lastFrame, setLastFrame] = useState<AudioFrameSummary | null>(null);
  const [logMessage, setLogMessage] = useState<string | null>(null);

  const frameSubRef = useRef<{ remove: () => void } | null>(null);
  const errorSubRef = useRef<{ remove: () => void } | null>(null);

  const handleFrame = useCallback((event: AudioFrameEvent) => {
    setFrameCount((prev) => prev + 1);

    const frameSummary: AudioFrameSummary = {
      sampleRate: event.sampleRate,
      length: event.pcmBase64.length,
      timestamp: event.timestamp,
      level: event.level ?? null,
    };

    setLastFrame(frameSummary);

    console.log("[AudioStreamTest] frame", {
      sampleRate: frameSummary.sampleRate,
      pcmBase64Length: frameSummary.length,
      timestamp: frameSummary.timestamp,
      level: frameSummary.level,
    });
  }, []);

  const detachListeners = useCallback(() => {
    frameSubRef.current?.remove();
    errorSubRef.current?.remove();
    frameSubRef.current = null;
    errorSubRef.current = null;
  }, []);

  const requestMicrophonePermission = useCallback(async () => {
    let permission = await requestPermission();

    if (Platform.OS === "android" && permission !== "granted") {
      const result = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      );
      permission =
        result === PermissionsAndroid.RESULTS.GRANTED ? "granted" : "denied";
    }

    if (Platform.OS === "ios" && permission === "undetermined") {
      permission = "granted";
    }

    return permission;
  }, []);

  const startStream = useCallback(async () => {
    if (isStarting || status === "recording") {
      return;
    }

    setIsStarting(true);
    setLogMessage(null);

    try {
      const permission = await requestMicrophonePermission();

      if (permission !== "granted") {
        setLogMessage(`Microphone permission not granted: ${permission}`);
        return;
      }

      setFrameCount(0);
      setLastFrame(null);
      detachListeners();

      frameSubRef.current = addFrameListener(handleFrame);
      errorSubRef.current = addErrorListener((event) => {
        setLogMessage(event.message);
        console.warn("[AudioStreamTest] error", event.message);
      });

      await start({
        sampleRate: 16000,
        frameDurationMs: 20,
        enableLevelMeter: true,
        enableBackground: true,
        enableBuffering: false,
      });

      const updatedStatus = await getStatus().catch(() => "recording");
      setStatus(updatedStatus);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      setLogMessage(`Failed to start stream: ${message}`);
      setStatus("idle");
    } finally {
      setIsStarting(false);
    }
  }, [
    detachListeners,
    handleFrame,
    isStarting,
    requestMicrophonePermission,
    status,
  ]);

  const stopStream = useCallback(async () => {
    detachListeners();

    try {
      await stop();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      setLogMessage(`Failed to stop stream: ${message}`);
    }

    const updatedStatus = await getStatus().catch(() => "stopped");
    setStatus(updatedStatus);
  }, [detachListeners]);

  const onStart = useCallback(() => {
    void startStream();
  }, [startStream]);

  const onStop = useCallback(() => {
    void stopStream();
  }, [stopStream]);

  useEffect(() => {
    getStatus()
      .then((currentStatus) => setStatus(currentStatus))
      .catch(() => setStatus("idle"));

    return () => {
      detachListeners();
      void stop();
    };
  }, [detachListeners]);

  return {
    status,
    isStarting,
    frameCount,
    lastFrame,
    logMessage,
    onStart,
    onStop,
  };
};

const AudioStreamTestView = ({
  status,
  isStarting,
  frameCount,
  lastFrame,
  logMessage,
  onStart,
  onStop,
}: AudioStreamTestViewProps) => {
  const isRecording = status === "recording";

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Audio Stream Test</Text>

      <View style={styles.card}>
        <View style={styles.row}>
          <Text style={styles.label}>Status</Text>
          <Text style={[styles.value, isRecording ? styles.good : styles.idle]}>
            {status}
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Frames</Text>
          <Text style={styles.value}>{frameCount}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Last frame</Text>
          <Text style={styles.value}>
            {lastFrame
              ? `${lastFrame.length} base64 chars • ${lastFrame.sampleRate} Hz`
              : "Waiting for frames"}
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Level</Text>
          <Text style={styles.value}>
            {lastFrame?.level ?? "n/a"}
          </Text>
        </View>
      </View>

      {logMessage ? <Text style={styles.error}>{logMessage}</Text> : null}

      <View style={styles.actions}>
        <Pressable
          onPress={onStart}
          disabled={isStarting || isRecording}
          style={({ pressed }) => [
            styles.button,
            styles.startButton,
            (isStarting || isRecording) && styles.buttonDisabled,
            pressed && !isRecording && !isStarting ? styles.buttonPressed : null,
          ]}
        >
          <Text style={styles.buttonText}>
            {isStarting ? "Starting..." : "Start"}
          </Text>
        </Pressable>
        <Pressable
          onPress={onStop}
          disabled={!isRecording}
          style={({ pressed }) => [
            styles.button,
            styles.stopButton,
            !isRecording && styles.buttonDisabled,
            pressed && isRecording ? styles.buttonPressed : null,
          ]}
        >
          <Text style={styles.buttonText}>Stop</Text>
        </Pressable>
      </View>
    </View>
  );
};

export const AudioStreamTest = () => {
  const viewModel = useAudioStreamTestViewModel();

  return <AudioStreamTestView {...viewModel} />;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 56,
    paddingBottom: 32,
    gap: 24,
    backgroundColor: "#0b0c10",
  },
  title: {
    fontSize: 26,
    fontWeight: "700",
    color: "#f5f5f5",
  },
  card: {
    padding: 20,
    borderRadius: 16,
    backgroundColor: "#161b22",
    gap: 12,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  label: {
    color: "#9aa4b2",
    fontSize: 14,
  },
  value: {
    color: "#f5f5f5",
    fontSize: 14,
    fontWeight: "600",
  },
  idle: {
    color: "#f5f5f5",
  },
  good: {
    color: "#5dd39e",
  },
  error: {
    color: "#ff6b6b",
    fontSize: 14,
  },
  actions: {
    flexDirection: "row",
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  startButton: {
    backgroundColor: "#2563eb",
  },
  stopButton: {
    backgroundColor: "#dc2626",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonPressed: {
    transform: [{ scale: 0.98 }],
  },
  buttonText: {
    color: "#ffffff",
    fontWeight: "600",
    fontSize: 16,
  },
});
