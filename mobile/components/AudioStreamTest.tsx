import { useEffect, useMemo, useRef, useState } from "react";
import {
  addErrorListener,
  addFrameListener,
  requestPermission,
  start,
  stop,
  type AudioFrameEvent,
} from "expo-stream-audio";
import {
  Button,
  PermissionsAndroid,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

type StreamStatus = "idle" | "recording" | "stopped" | "error";
type PermissionStatus = "unknown" | "granted" | "denied" | "undetermined";

type StreamConfig = {
  sampleRate: 16000 | 44100 | 48000;
  frameDurationMs: number;
  enableLevelMeter: boolean;
};

type FrameMetrics = {
  base64Length: number;
  byteLength: number;
  sampleCount: number;
  durationMs: number;
  timeSinceLastFrameMs: number;
  framesPerSecond: number;
};

const STREAM_CONFIG: StreamConfig = {
  sampleRate: 16000,
  frameDurationMs: 20,
  enableLevelMeter: true,
};

const getBase64ByteLength = (base64: string): number => {
  if (base64.length === 0) {
    return 0;
  }

  const normalized = base64.replace(/\s/g, "");
  const padding = normalized.endsWith("==")
    ? 2
    : normalized.endsWith("=")
      ? 1
      : 0;
  return Math.max(0, Math.floor((normalized.length * 3) / 4) - padding);
};

const formatNumber = (value: number, digits: number): string =>
  value.toFixed(digits);

const formatTimestamp = (timestamp: number): string =>
  new Date(timestamp).toLocaleTimeString();

const formatDuration = (valueMs: number): string =>
  valueMs >= 1000
    ? `${formatNumber(valueMs / 1000, 2)}s`
    : `${Math.round(valueMs)}ms`;

const normalizePermissionStatus = (value: string): PermissionStatus => {
  switch (value) {
    case "granted":
    case "denied":
    case "undetermined":
      return value;
    default:
      return "unknown";
  }
};

export function MicStreamTest() {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [permissionStatus, setPermissionStatus] =
    useState<PermissionStatus>("unknown");
  const [lastFrame, setLastFrame] = useState<AudioFrameEvent | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const frameSubRef = useRef<{ remove: () => void } | null>(null);
  const errorSubRef = useRef<{ remove: () => void } | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      frameSubRef.current?.remove();
      errorSubRef.current?.remove();
      stop().catch(() => {});
    };
  }, []);

  useEffect(() => {
    if (status !== "recording") {
      return;
    }

    const interval = setInterval(() => {
      setTick((value) => value + 1);
    }, 500);

    return () => clearInterval(interval);
  }, [status]);

  const handleStart = async () => {
    let permission = await requestPermission();

    if (Platform.OS === "android" && permission !== "granted") {
      const result = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      );
      if (result === PermissionsAndroid.RESULTS.GRANTED) {
        permission = "granted";
      }
    }

    if (Platform.OS === "ios" && permission === "undetermined") {
      // Allow native code to show the system mic prompt.
      permission = "granted";
    }

    setPermissionStatus(normalizePermissionStatus(permission));

    if (permission !== "granted") {
      return;
    }

    frameSubRef.current?.remove();
    errorSubRef.current?.remove();
    setLastFrame(null);
    setFrameCount(0);
    setLastError(null);
    setTick(0);
    startTimeRef.current = Date.now();
    lastFrameTimeRef.current = null;

    frameSubRef.current = addFrameListener((frame: AudioFrameEvent) => {
      setLastFrame(frame);
      setFrameCount((count) => count + 1);
      lastFrameTimeRef.current = frame.timestamp;
    });

    errorSubRef.current = addErrorListener((event) => {
      console.warn("Mic error", event.message);
      setLastError(event.message);
      setStatus("error");
    });

    try {
      await start(STREAM_CONFIG);
      setStatus("recording");
    } catch (error) {
      console.warn("Mic start error", error);
      setLastError("Failed to start the microphone stream.");
      setStatus("error");
    }
  };

  const handleStop = async () => {
    frameSubRef.current?.remove();
    errorSubRef.current?.remove();
    await stop();
    setStatus("stopped");
  };

  const statusMeta = useMemo(() => {
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
  }, [status]);

  const levelValue = useMemo(() => {
    if (!lastFrame) {
      return null;
    }

    if (typeof lastFrame.level !== "number") {
      return null;
    }

    if (!Number.isFinite(lastFrame.level)) {
      return null;
    }

    return lastFrame.level;
  }, [lastFrame]);

  const frameMetrics = useMemo<FrameMetrics>(() => {
    const base64Length = lastFrame ? lastFrame.pcmBase64.length : 0;
    const byteLength = lastFrame
      ? getBase64ByteLength(lastFrame.pcmBase64)
      : 0;
    const sampleCount = Math.floor(byteLength / 2);
    const durationMs =
      sampleCount > 0
        ? (sampleCount / STREAM_CONFIG.sampleRate) * 1000
        : 0;
    const now = Date.now();
    const timeSinceLastFrameMs =
      lastFrameTimeRef.current !== null ? now - lastFrameTimeRef.current : 0;
    const startTime = startTimeRef.current;
    const elapsedMs = startTime !== null ? now - startTime : 0;
    const framesPerSecond =
      elapsedMs > 0 ? (frameCount / elapsedMs) * 1000 : 0;

    return {
      base64Length,
      byteLength,
      sampleCount,
      durationMs,
      timeSinceLastFrameMs,
      framesPerSecond,
    };
  }, [frameCount, lastFrame, tick]);

  const expectedSamples = Math.round(
    (STREAM_CONFIG.sampleRate * STREAM_CONFIG.frameDurationMs) / 1000,
  );
  const expectedBytes = expectedSamples * 2;

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView
        contentContainerStyle={styles.content}
        alwaysBounceVertical={false}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Mic Stream Monitor</Text>
          <Text style={styles.subtitle}>
            Live snapshot of frame timing, payload size, and level meter.
          </Text>
        </View>

        <View style={styles.row}>
          <View style={[styles.pill, { backgroundColor: statusMeta.bg }]}>
            <View
              style={[styles.statusDot, { backgroundColor: statusMeta.color }]}
            />
            <Text style={[styles.pillText, { color: statusMeta.color }]}>
              {statusMeta.label}
            </Text>
          </View>
          <View style={styles.pillMuted}>
            <Text style={styles.pillMutedText}>
              Permission: {permissionStatus}
            </Text>
          </View>
        </View>

        <View style={styles.controls}>
          <View style={styles.buttonWrap}>
            <Button title="Start Stream" onPress={handleStart} />
          </View>
          <View style={styles.buttonWrap}>
            <Button title="Stop Stream" onPress={handleStop} />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Session Settings</Text>
          <View style={styles.grid}>
            <View style={styles.gridItem}>
              <Text style={styles.label}>Target sample rate</Text>
              <Text style={styles.value}>
                {STREAM_CONFIG.sampleRate} Hz
              </Text>
            </View>
            <View style={styles.gridItem}>
              <Text style={styles.label}>Frame duration</Text>
              <Text style={styles.value}>
                {STREAM_CONFIG.frameDurationMs} ms
              </Text>
            </View>
            <View style={styles.gridItem}>
              <Text style={styles.label}>Expected samples</Text>
              <Text style={styles.value}>{expectedSamples}</Text>
            </View>
            <View style={styles.gridItem}>
              <Text style={styles.label}>Expected bytes</Text>
              <Text style={styles.value}>{expectedBytes}</Text>
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Frame Snapshot</Text>
          {lastFrame ? (
            <View style={styles.grid}>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Frame sample rate</Text>
                <Text style={styles.value}>{lastFrame.sampleRate} Hz</Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Timestamp</Text>
                <Text style={styles.value}>
                  {formatTimestamp(lastFrame.timestamp)}
                </Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>PCM base64 length</Text>
                <Text style={styles.value}>
                  {frameMetrics.base64Length}
                </Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>PCM bytes</Text>
                <Text style={styles.value}>{frameMetrics.byteLength}</Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Samples in frame</Text>
                <Text style={styles.value}>{frameMetrics.sampleCount}</Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Frame duration</Text>
                <Text style={styles.value}>
                  {formatDuration(frameMetrics.durationMs)}
                </Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Frames/sec</Text>
                <Text style={styles.value}>
                  {formatNumber(frameMetrics.framesPerSecond, 2)}
                </Text>
              </View>
              <View style={styles.gridItem}>
                <Text style={styles.label}>Time since frame</Text>
                <Text style={styles.value}>
                  {formatDuration(frameMetrics.timeSinceLastFrameMs)}
                </Text>
              </View>
            </View>
          ) : (
            <Text style={styles.emptyText}>No frames yet.</Text>
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Level Meter</Text>
          <View style={styles.meter}>
            <View
              style={[
                styles.meterFill,
                {
                  width: `${Math.min(
                    100,
                    Math.max(0, (levelValue ?? 0) * 100),
                  )}%`,
                },
              ]}
            />
          </View>
          <Text style={styles.meterValue}>
            {levelValue === null
              ? "Level unavailable"
              : `Level: ${formatNumber(levelValue, 3)}`}
          </Text>
          <Text style={styles.helper}>
            Frames received: {frameCount}
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Diagnostics</Text>
          <Text style={styles.helper}>
            Listener status:{" "}
            {status === "recording" ? "active" : "inactive"}
          </Text>
          <Text style={styles.helper}>
            Last error: {lastError ?? "None"}
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

export const AudioStreamTest = MicStreamTest;

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F8FAFC",
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
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  gridItem: {
    minWidth: "45%",
    flex: 1,
    gap: 4,
  },
  label: {
    fontSize: 12,
    color: "#64748B",
  },
  value: {
    fontSize: 15,
    fontWeight: "600",
    color: "#0F172A",
  },
  emptyText: {
    fontSize: 14,
    color: "#94A3B8",
  },
  meter: {
    height: 14,
    borderRadius: 999,
    backgroundColor: "#E2E8F0",
    overflow: "hidden",
  },
  meterFill: {
    height: "100%",
    borderRadius: 999,
    backgroundColor: "#0EA5E9",
  },
  meterValue: {
    fontSize: 14,
    fontWeight: "600",
    color: "#0F172A",
  },
  helper: {
    fontSize: 12,
    color: "#64748B",
  },
});
