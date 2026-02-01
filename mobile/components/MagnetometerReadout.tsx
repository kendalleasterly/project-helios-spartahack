import { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { Magnetometer } from "expo-sensors";

type MagnetometerSubscription = ReturnType<typeof Magnetometer.addListener>;

type MagnetometerState = {
  x: number;
  z: number;
};

const DEFAULT_MAGNETOMETER: MagnetometerState = {
  x: 0,
  z: 0,
};

export function MagnetometerReadout() {
  const [magnetometer, setMagnetometer] = useState<MagnetometerState>(
    DEFAULT_MAGNETOMETER
  );

  useEffect(() => {
    let subscription: MagnetometerSubscription | null = null;
    let isMounted = true;

    const start = async () => {
      const isAvailable = await Magnetometer.isAvailableAsync();
      if (!isAvailable) {
        console.warn("Magnetometer not available on this device.");
        return;
      }

      const permission = await Magnetometer.getPermissionsAsync();
      if (!permission.granted) {
        const request = await Magnetometer.requestPermissionsAsync();
        if (!request.granted) {
          console.warn("Magnetometer permission not granted.");
          return;
        }
      }

      if (!isMounted) {
        return;
      }

      Magnetometer.setUpdateInterval(50);
      subscription = Magnetometer.addListener(({ x, z }) => {
        setMagnetometer({
          x: Number.isFinite(x) ? x : 0,
          z: Number.isFinite(z) ? z : 0,
        });
      });
    };

    start();

    return () => {
      isMounted = false;
      if (subscription) {
        subscription.remove();
      }
    };
  }, []);

  const safeX = Number.isFinite(magnetometer.x) ? magnetometer.x : 0;
  const safeZ = Number.isFinite(magnetometer.z) ? magnetometer.z : 0;

  return (
    <View pointerEvents="none" style={styles.container}>
      <Text style={styles.title}>Magnetometer (μT)</Text>
      <View style={styles.row}>
        <Text style={styles.label}>x</Text>
        <Text style={styles.value}>{safeX.toFixed(2)}</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>z</Text>
        <Text style={styles.value}>{safeZ.toFixed(2)}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignSelf: "stretch",
    width: "100%",
    backgroundColor: "rgba(15, 23, 42, 0.75)",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 12,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
    alignItems: "center",
  },
  title: {
    fontSize: 12,
    fontWeight: "700",
    color: "#E2E8F0",
    letterSpacing: 0.2,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },
  label: {
    fontSize: 11,
    color: "#CBD5F5",
    fontWeight: "600",
  },
  value: {
    fontSize: 12,
    color: "#F8FAFC",
    fontWeight: "700",
    textAlign: "right",
    minWidth: 88,
  },
});
