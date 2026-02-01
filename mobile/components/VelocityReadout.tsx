import { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { DeviceMotion } from "expo-sensors";

type DeviceMotionSubscription = ReturnType<typeof DeviceMotion.addListener>;

type VelocityAverages = {
  vx: number;
  vz: number;
  speed: number;
};

type VelocityState = {
  vx: number;
  vz: number;
  speed: number;
  horizontalDir: number;
  avg200: VelocityAverages;
  avg500: VelocityAverages;
};

const DEFAULT_VELOCITY: VelocityState = {
  vx: 0,
  vz: 0,
  speed: 0,
  horizontalDir: 0,
  avg200: { vx: 0, vz: 0, speed: 0 },
  avg500: { vx: 0, vz: 0, speed: 0 },
};

const DRAG = 0.98;
const STILL_ACCEL_THRESHOLD = 0.15;
const STILL_ROTATION_THRESHOLD = 5;
const MIN_DT = 0.001;
const MAX_DT = 0.1;

const computeAverages = (
  samples: Array<{ t: number; vx: number; vz: number; speed: number }>,
  cutoff: number
): VelocityAverages => {
  let sumVx = 0;
  let sumVz = 0;
  let sumSpeed = 0;
  let count = 0;

  for (let i = samples.length - 1; i >= 0; i -= 1) {
    const sample = samples[i];
    if (sample.t < cutoff) {
      break;
    }
    sumVx += sample.vx;
    sumVz += sample.vz;
    sumSpeed += sample.speed;
    count += 1;
  }

  if (count === 0) {
    return { vx: 0, vz: 0, speed: 0 };
  }

  return {
    vx: sumVx / count,
    vz: sumVz / count,
    speed: sumSpeed / count,
  };
};

export function VelocityReadout() {
  const [velocity, setVelocity] = useState<VelocityState>(DEFAULT_VELOCITY);
  const velocityRef = useRef({
    vx: 0,
    vz: 0,
  });
  const lastTRef = useRef<number | null>(null);
  const samplesRef = useRef<
    Array<{ t: number; vx: number; vz: number; speed: number }>
  >([]);

  useEffect(() => {
    let subscription: DeviceMotionSubscription | null = null;
    let isMounted = true;

    const start = async () => {
      const isAvailable = await DeviceMotion.isAvailableAsync();
      if (!isAvailable) {
        console.warn("Device motion not available on this device.");
        return;
      }

      const permission = await DeviceMotion.getPermissionsAsync();
      if (!permission.granted) {
        const request = await DeviceMotion.requestPermissionsAsync();
        if (!request.granted) {
          console.warn("Device motion permission not granted.");
          return;
        }
      }

      if (!isMounted) {
        return;
      }

      // Faster sampling for iOS; adjust as needed for stability/noise.
      DeviceMotion.setUpdateInterval(50);

      subscription = DeviceMotion.addListener(({ acceleration, rotationRate }) => {
        if (!acceleration) {
          return;
        }

        const timestampSeconds =
          acceleration.timestamp > 1_000_000
            ? acceleration.timestamp / 1000
            : acceleration.timestamp;
        const lastT = lastTRef.current;
        if (lastT === null) {
          lastTRef.current = timestampSeconds;
          return;
        }

        const dt = Math.min(
          MAX_DT,
          Math.max(MIN_DT, timestampSeconds - lastT)
        );
        lastTRef.current = timestampSeconds;

        let { vx, vz } = velocityRef.current;

        vx += acceleration.x * dt;
        vz += acceleration.z * dt;

        vx *= DRAG;
        vz *= DRAG;

        const aMag = Math.hypot(
          acceleration.x,
          acceleration.y,
          acceleration.z
        );
        const rMag = rotationRate
          ? Math.hypot(
              rotationRate.alpha,
              rotationRate.beta,
              rotationRate.gamma
            )
          : 0;
        const isStill =
          aMag < STILL_ACCEL_THRESHOLD && rMag < STILL_ROTATION_THRESHOLD;

        if (isStill) {
          vx = 0;
          vz = 0;
        }

        velocityRef.current = { vx, vz };

        const speed = Math.hypot(vx, vz);
        const horizontalDir = Math.atan2(vx, vz);

        const samples = samplesRef.current;
        samples.push({ t: timestampSeconds, vx, vz, speed });

        const cutoff500 = timestampSeconds - 0.5;
        while (samples.length > 0 && samples[0].t < cutoff500) {
          samples.shift();
        }

        const avg200 = computeAverages(samples, timestampSeconds - 0.2);
        const avg500 = computeAverages(samples, cutoff500);

        setVelocity({ vx, vz, speed, horizontalDir, avg200, avg500 });
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

  return (
    <View pointerEvents="none" style={styles.container}>
      <Text style={styles.title}>Velocity (device frame)</Text>
      <View style={styles.row}>
        <Text style={styles.label}>vx</Text>
        <Text style={styles.value}>{(velocity.vx * 100).toFixed(1)} cm/s</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>vz</Text>
        <Text style={styles.value}>{(velocity.vz * 100).toFixed(1)} cm/s</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>speed</Text>
        <Text style={styles.value}>{(velocity.speed * 100).toFixed(1)} cm/s</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>dir</Text>
        <Text style={styles.value}>
          {(velocity.horizontalDir * (180 / Math.PI)).toFixed(1)}°
        </Text>
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Avg 200ms</Text>
        <View style={styles.row}>
          <Text style={styles.label}>vx</Text>
          <Text style={styles.value}>
            {(velocity.avg200.vx * 100).toFixed(1)} cm/s
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>vz</Text>
          <Text style={styles.value}>
            {(velocity.avg200.vz * 100).toFixed(1)} cm/s
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>speed</Text>
          <Text style={styles.value}>
            {(velocity.avg200.speed * 100).toFixed(1)} cm/s
          </Text>
        </View>
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Avg 500ms</Text>
        <View style={styles.row}>
          <Text style={styles.label}>vx</Text>
          <Text style={styles.value}>
            {(velocity.avg500.vx * 100).toFixed(1)} cm/s
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>vz</Text>
          <Text style={styles.value}>
            {(velocity.avg500.vz * 100).toFixed(1)} cm/s
          </Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>speed</Text>
          <Text style={styles.value}>
            {(velocity.avg500.speed * 100).toFixed(1)} cm/s
          </Text>
        </View>
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
  section: {
    alignSelf: "stretch",
    gap: 8,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "700",
    color: "#E2E8F0",
    letterSpacing: 0.2,
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
