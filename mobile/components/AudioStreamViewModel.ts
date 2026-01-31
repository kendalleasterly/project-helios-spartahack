import { useEffect, useMemo, useRef, useState } from "react";
import {
  addErrorListener,
  addFrameListener,
  requestPermission,
  start,
  stop,
  type AudioFrameEvent,
} from "expo-stream-audio";
import { PermissionsAndroid, Platform } from "react-native";
import { toByteArray } from "base64-js";
import {
  DeepgramRawClient,
  type DeepgramAuthMode,
  type DeepgramConfig,
  type DeepgramEvent,
} from "@/components/deepgram/DeepgramRawClient";

export type StreamStatus = "idle" | "recording" | "stopped" | "error";
export type PermissionStatus = "unknown" | "granted" | "denied" | "undetermined";
export type DeepgramStatus = "idle" | "connecting" | "open" | "closed" | "error";

export type FrameMetrics = {
  base64Length: number;
  byteLength: number;
  sampleCount: number;
  durationMs: number;
  timeSinceLastFrameMs: number;
  framesPerSecond: number;
};

export type AudioStreamViewState = {
  status: StreamStatus;
  permissionStatus: PermissionStatus;
  deepgramStatus: DeepgramStatus;
  deepgramError: string | null;
  deepgramSampleRate: number | null;
  deepgramAuthMode: DeepgramAuthMode;
  lastFrame: AudioFrameEvent | null;
  frameCount: number;
  frameMetrics: FrameMetrics;
  levelValue: number | null;
  expectedSamples: number;
  expectedBytes: number;
  sessionSampleRate: number;
  sessionChannels: number;
  sessionFrameDurationMs: number;
  partialTranscript: string;
  finalTranscripts: string[];
  micError: string | null;
};

export type AudioStreamViewActions = {
  onStart: () => void;
  onStop: () => void;
};

type StreamConfig = {
  sampleRate: 16000 | 44100 | 48000;
  channels: 1;
  frameDurationMs: number;
  enableLevelMeter: boolean;
};

type DeepgramTranscriptEvent = {
  type: "Results";
  transcript: string;
  isFinal: boolean;
  speechFinal: boolean;
  fromFinalize: boolean;
};

const STREAM_CONFIG: StreamConfig = {
  sampleRate: 16000,
  channels: 1,
  frameDurationMs: 20,
  enableLevelMeter: true,
};

const DEFAULT_DEEPGRAM_CONFIG = {
  model: "nova-3",
  language: "en-US",
  interimResults: true,
  endpointingMs: 300,
  utteranceEndMs: 1000,
  vadEvents: true,
  punctuate: false,
  smartFormat: false,
  noDelay: true,
  keepAliveMs: 4000,
  enableKeepAlive: true,
};

const sanitizeApiKey = (value: string): string => {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
};

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

const normalizeTranscriptEvent = (value: unknown): DeepgramTranscriptEvent | null => {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  if (record.type !== "Results") {
    return null;
  }

  const channelValue = record.channel;
  if (!channelValue || typeof channelValue !== "object") {
    return null;
  }

  const alternativesValue = (channelValue as Record<string, unknown>).alternatives;
  if (!Array.isArray(alternativesValue) || alternativesValue.length === 0) {
    return null;
  }

  const firstAlternative = alternativesValue[0];
  if (!firstAlternative || typeof firstAlternative !== "object") {
    return null;
  }

  const transcriptValue = (firstAlternative as Record<string, unknown>).transcript;
  if (typeof transcriptValue !== "string") {
    return null;
  }

  return {
    type: "Results",
    transcript: transcriptValue,
    isFinal: record.is_final === true,
    speechFinal: record.speech_final === true,
    fromFinalize: record.from_finalize === true,
  };
};

const trimTranscripts = (entries: string[], limit: number): string[] =>
  entries.length > limit ? entries.slice(entries.length - limit) : entries;

export const useAudioStreamViewModel = (): {
  state: AudioStreamViewState;
  actions: AudioStreamViewActions;
} => {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [permissionStatus, setPermissionStatus] =
    useState<PermissionStatus>("unknown");
  const [deepgramStatus, setDeepgramStatus] =
    useState<DeepgramStatus>("idle");
  const [lastFrame, setLastFrame] = useState<AudioFrameEvent | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);
  const [deepgramError, setDeepgramError] = useState<string | null>(null);
  const [deepgramSampleRate, setDeepgramSampleRate] = useState<number | null>(null);
  const [partialTranscript, setPartialTranscript] = useState("");
  const [finalTranscripts, setFinalTranscripts] = useState<string[]>([]);
  const [tick, setTick] = useState(0);

  const frameSubRef = useRef<{ remove: () => void } | null>(null);
  const errorSubRef = useRef<{ remove: () => void } | null>(null);
  const deepgramClientRef = useRef<DeepgramRawClient | null>(null);
  const deepgramPendingFramesRef = useRef<ArrayBuffer[]>([]);
  const deepgramStopRequestedRef = useRef(false);
  const deepgramStartInFlightRef = useRef(false);
  const deepgramReadyRef = useRef(false);
  const deepgramSampleRateRef = useRef<number | null>(null);
  const audioFrameLogRef = useRef(0);
  const startTimeRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number | null>(null);

  const apiKey = useMemo(
    () => sanitizeApiKey(process.env.EXPO_PUBLIC_DEEPGRAM_SECRET ?? ""),
    [],
  );

  const deepgramAuthMode: DeepgramAuthMode =
    Platform.OS === "web" ? "token" : "header";

  const stopDeepgram = () => {
    deepgramStopRequestedRef.current = true;
    deepgramStartInFlightRef.current = false;
    deepgramReadyRef.current = false;
    deepgramPendingFramesRef.current = [];

    const client = deepgramClientRef.current;
    deepgramClientRef.current = null;

    if (client) {
      client.finalizeAndClose();
      client.close();
    }

    setDeepgramStatus("closed");
  };

  const handleDeepgramEvent = (event: DeepgramEvent) => {
    switch (event.type) {
      case "open": {
        console.log("Deepgram open");
        deepgramReadyRef.current = true;
        deepgramStartInFlightRef.current = false;
        setDeepgramStatus("open");

        const queuedFrames = deepgramPendingFramesRef.current.splice(0);
        queuedFrames.forEach((buffer) => {
          deepgramClientRef.current?.sendAudio(buffer);
        });
        break;
      }
      case "close": {
        console.log("Deepgram closed", event.code, event.reason);
        deepgramReadyRef.current = false;
        deepgramClientRef.current = null;
        deepgramStartInFlightRef.current = false;
        setDeepgramStatus("closed");

        if (!deepgramStopRequestedRef.current && event.code !== 1000) {
          const reason = event.reason ? ` (${event.reason})` : "";
          setDeepgramError(`Deepgram closed: ${event.code}${reason}`);
        }
        break;
      }
      case "error": {
        console.log("Deepgram error event", event.message);
        deepgramStartInFlightRef.current = false;
        setDeepgramStatus("error");
        setDeepgramError(event.message);
        break;
      }
      case "message": {
        console.log("Deepgram message", event.raw);
        if (!event.parsed) {
          return;
        }

        const transcriptEvent = normalizeTranscriptEvent(event.parsed);
        if (!transcriptEvent) {
          return;
        }

        const text = transcriptEvent.transcript.trim();
        if (!text) {
          return;
        }

        if (transcriptEvent.isFinal || transcriptEvent.speechFinal || transcriptEvent.fromFinalize) {
          setFinalTranscripts((entries) => trimTranscripts([...entries, text], 12));
          setPartialTranscript("");
        } else {
          setPartialTranscript(text);
        }
        break;
      }
    }
  };

  const startDeepgram = async (sampleRate: number) => {
    if (!apiKey) {
      const message = "Missing Deepgram API key.";
      setDeepgramStatus("error");
      setDeepgramError(message);
      throw new Error(message);
    }

    stopDeepgram();
    setDeepgramStatus("connecting");
    setDeepgramError(null);
    setPartialTranscript("");
    setFinalTranscripts([]);
    deepgramStopRequestedRef.current = false;
    deepgramStartInFlightRef.current = true;
    deepgramSampleRateRef.current = sampleRate;
    setDeepgramSampleRate(sampleRate);

    const config: DeepgramConfig = {
      apiKey,
      sampleRate,
      channels: STREAM_CONFIG.channels,
      authMode: deepgramAuthMode,
      ...DEFAULT_DEEPGRAM_CONFIG,
    };

    const client = new DeepgramRawClient(config, { onEvent: handleDeepgramEvent });
    deepgramClientRef.current = client;

    await client.connect();
  };

  useEffect(() => {
    return () => {
      frameSubRef.current?.remove();
      errorSubRef.current?.remove();
      stopDeepgram();
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
    setMicError(null);
    setDeepgramError(null);
    setTick(0);
    startTimeRef.current = Date.now();
    lastFrameTimeRef.current = null;
    deepgramSampleRateRef.current = null;
    setDeepgramSampleRate(null);
    deepgramPendingFramesRef.current = [];
    audioFrameLogRef.current = 0;

    frameSubRef.current = addFrameListener((frame: AudioFrameEvent) => {
      setLastFrame(frame);
      setFrameCount((count) => count + 1);
      lastFrameTimeRef.current = frame.timestamp;

      const bytes = toByteArray(frame.pcmBase64);
      if (bytes.byteLength === 0) {
        return;
      }

      const audioBuffer = bytes.slice().buffer;

      audioFrameLogRef.current += 1;
      if (audioFrameLogRef.current <= 3 || audioFrameLogRef.current % 50 === 0) {
        console.log("Deepgram audio frame", {
          count: audioFrameLogRef.current,
          sampleRate: frame.sampleRate,
          bytes: bytes.byteLength,
        });
      }

      if (deepgramSampleRateRef.current === null) {
        deepgramSampleRateRef.current = frame.sampleRate;
        setDeepgramSampleRate(frame.sampleRate);
        deepgramPendingFramesRef.current.push(audioBuffer);
        if (!deepgramStartInFlightRef.current) {
          startDeepgram(frame.sampleRate).catch((error) => {
            console.warn("Deepgram start error", error);
            setMicError("Failed to start Deepgram.");
            setStatus("error");
          });
        }
        return;
      }

      if (deepgramSampleRateRef.current !== frame.sampleRate) {
        deepgramSampleRateRef.current = frame.sampleRate;
        setDeepgramSampleRate(frame.sampleRate);
        deepgramPendingFramesRef.current = [audioBuffer];
        if (!deepgramStartInFlightRef.current) {
          startDeepgram(frame.sampleRate).catch((error) => {
            console.warn("Deepgram restart error", error);
          });
        }
        return;
      }

      if (!deepgramReadyRef.current || !deepgramClientRef.current?.isOpen()) {
        if (deepgramPendingFramesRef.current.length < 10) {
          deepgramPendingFramesRef.current.push(audioBuffer);
        }
        return;
      }

      deepgramClientRef.current.sendAudio(audioBuffer);
    });

    errorSubRef.current = addErrorListener((event) => {
      console.warn("Mic error", event.message);
      setMicError(event.message);
      setStatus("error");
    });

    try {
      await start(STREAM_CONFIG);
      setStatus("recording");
    } catch (error) {
      console.warn("Mic start error", error);
      setMicError("Failed to start the microphone stream.");
      stopDeepgram();
      setStatus("error");
    }
  };

  const handleStop = async () => {
    frameSubRef.current?.remove();
    errorSubRef.current?.remove();
    try {
      await stop();
    } catch {}
    stopDeepgram();
    setStatus("stopped");
  };

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
    const byteLength = lastFrame ? getBase64ByteLength(lastFrame.pcmBase64) : 0;
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
    const framesPerSecond = elapsedMs > 0 ? (frameCount / elapsedMs) * 1000 : 0;

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

  const state: AudioStreamViewState = {
    status,
    permissionStatus,
    deepgramStatus,
    deepgramError,
    deepgramSampleRate,
    deepgramAuthMode,
    lastFrame,
    frameCount,
    frameMetrics,
    levelValue,
    expectedSamples,
    expectedBytes,
    sessionSampleRate: STREAM_CONFIG.sampleRate,
    sessionChannels: STREAM_CONFIG.channels,
    sessionFrameDurationMs: STREAM_CONFIG.frameDurationMs,
    partialTranscript,
    finalTranscripts,
    micError,
  };

  const actions: AudioStreamViewActions = {
    onStart: handleStart,
    onStop: handleStop,
  };

  return { state, actions };
};
