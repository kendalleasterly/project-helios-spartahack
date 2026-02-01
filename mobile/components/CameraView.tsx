import { useEffect, useState, useRef, useCallback } from "react";
import { StyleSheet, View, Text, ActivityIndicator } from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";

const FRAME_RATE = 10;
const TARGET_WIDTH = 720;  // 720p for Gemini quality
const TARGET_HEIGHT = 1280;

type CameraViewProps = {
  onFrame: (base64Frame: string, userQuestion?: string, debug?: boolean) => void;
  getPendingQuestion?: () => string | undefined;
  onFrameCaptured?: (base64Frame: string) => void;
};

export default function CameraView({ onFrame, getPendingQuestion, onFrameCaptured }: CameraViewProps) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const device = useCameraDevice("back");
  const camera = useRef<Camera>(null);
  const captureInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const isCapturing = useRef(false);

  // Use refs to always have access to latest callbacks (avoids stale closure in setInterval)
  const onFrameRef = useRef(onFrame);
  const getPendingQuestionRef = useRef(getPendingQuestion);
  const onFrameCapturedRef = useRef(onFrameCaptured);

  // Keep refs up to date
  useEffect(() => {
    onFrameRef.current = onFrame;
    getPendingQuestionRef.current = getPendingQuestion;
    onFrameCapturedRef.current = onFrameCaptured;
  }, [onFrame, getPendingQuestion, onFrameCaptured]);

  const captureFrame = useCallback(async () => {
    if (camera.current == null || isCapturing.current) return;

    try {
      isCapturing.current = true;
      const photo = await camera.current.takePhoto({
        enableShutterSound: false,
      });

      // Downscale to 720p (720x1280 for portrait after EXIF rotation)
      // manipulateAsync auto-rotates based on EXIF, so resize to final display size
      const resized = await manipulateAsync(
        photo.path,
        [{ resize: { width: TARGET_WIDTH, height: TARGET_HEIGHT } }],
        { compress: 0.7, format: SaveFormat.JPEG }
      );

      const response = await fetch(resized.uri);
      const blob = await response.blob();
      const reader = new FileReader();

      reader.onloadend = () => {
        const base64 = reader.result as string;
        const base64Data = base64.split(",")[1];

        // Notify about the captured frame (for person memory feature)
        onFrameCapturedRef.current?.(base64Data);

        // Check for pending question from wake word detection
        const pendingQuestion = getPendingQuestionRef.current?.();
        
        if (pendingQuestion) {
          console.log(
            `Frame captured with question: ${photo.width}x${photo.height} → ${resized.width}x${resized.height}, question: "${pendingQuestion}"`,
          );
        } else {
          // console.log(
          //   `Frame captured: ${photo.width}x${photo.height} → ${resized.width}x${resized.height}, base64 length: ${base64Data.length}`,
          // );
        }

        // Enable debug mode to see bounding boxes on frames
        const DEBUG_MODE = true;
        onFrameRef.current(base64Data, pendingQuestion, DEBUG_MODE);
      };

      reader.readAsDataURL(blob);
    } catch (error) {
      console.error("Error capturing frame:", error);
    } finally {
      isCapturing.current = false;
    }
  }, []); // No dependencies - uses refs for latest values

  useEffect(() => {
    if (hasPermission === false) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    setIsActive(true);
    return () => {
      setIsActive(false);
    };
  }, []);

  useEffect(() => {
    if (isActive && hasPermission && device != null) {
      const timeout = setTimeout(() => {
        captureInterval.current = setInterval(captureFrame, 1000 / FRAME_RATE);
      }, 0); // timeout could be 2000 to wait for camera to stabilize if needed

      return () => {
        clearTimeout(timeout);
        if (captureInterval.current) {
          clearInterval(captureInterval.current);
          captureInterval.current = null;
        }
      };
    }
  }, [isActive, hasPermission, device, captureFrame]);

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Camera permission is required</Text>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" />
        <Text style={styles.text}>Loading camera...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        photo={true}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  text: {
    color: "#fff",
    fontSize: 16,
    marginTop: 10,
  },
});
