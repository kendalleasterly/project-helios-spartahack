import { useEffect, useState, useRef, useCallback } from "react";
import { StyleSheet, View, Text, ActivityIndicator } from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";

const FRAME_RATE = 1;

type CameraViewProps = {
  onFrame: (base64Frame: string, debug?: boolean) => void;
};

export default function CameraView({ onFrame }: CameraViewProps) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const device = useCameraDevice("back");
  const camera = useRef<Camera>(null);
  const captureInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const captureFrame = useCallback(async () => {
    if (camera.current == null) return;

    try {
      const photo = await camera.current.takePhoto({
        enableShutterSound: false,
      });

      const response = await fetch(`file://${photo.path}`);
      const blob = await response.blob();
      const reader = new FileReader();

      reader.onloadend = () => {
        const base64 = reader.result as string;
        const base64Data = base64.split(",")[1];
        console.log(
          `Frame captured: ${photo.width}x${photo.height}, base64 JPEG length: ${base64Data.length}`,
        );

        onFrame(base64Data);
      };

      reader.readAsDataURL(blob);
    } catch (error) {
      console.error("Error capturing frame:", error);
    }
  }, [onFrame]);

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
