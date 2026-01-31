import { useEffect, useRef, useState } from "react";
import {
  addErrorListener,
  addFrameListener,
  requestPermission,
  start,
  stop,
  type AudioFrameEvent,
} from "expo-stream-audio";
import { Button, PermissionsAndroid, Platform, Text, View } from "react-native";

export function MicStreamTest() {
  const [status, setStatus] = useState("idle");
  const [lastFrame, setLastFrame] = useState<AudioFrameEvent | null>(null);
  const frameSubRef = useRef<{ remove: () => void } | null>(null);
  const errorSubRef = useRef<{ remove: () => void } | null>(null);

  useEffect(() => {
    return () => {
      frameSubRef.current?.remove();
      errorSubRef.current?.remove();
      stop().catch(() => {});
    };
  }, []);

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

    if (permission !== "granted") {
      return;
    }

    frameSubRef.current?.remove();
    errorSubRef.current?.remove();

    frameSubRef.current = addFrameListener((frame: AudioFrameEvent) => {
      setLastFrame(frame);
    });

    errorSubRef.current = addErrorListener((event) => {
      console.warn("Mic error", event.message);
    });

    await start({
      sampleRate: 16000,
      frameDurationMs: 20,
      enableLevelMeter: true,
    });
    setStatus("recording");
  };

  const handleStop = async () => {
    frameSubRef.current?.remove();
    errorSubRef.current?.remove();
    await stop();
    setStatus("stopped");
  };

  return (
    <View>
      <Text>Status: {status}</Text>
      <Button title="Start" onPress={handleStart} />
      <Button title="Stop" onPress={handleStop} />
      {lastFrame ? (
        <Text>
          {lastFrame.sampleRate} Hz - level=
          {lastFrame.level?.toFixed(6) ?? "n/a"}
        </Text>
      ) : (
        <Text>No frames yet</Text>
      )}
    </View>
  );
}

export const AudioStreamTest = MicStreamTest;
