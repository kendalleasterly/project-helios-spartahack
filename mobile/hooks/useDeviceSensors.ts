import { useEffect, useRef, useState } from "react";
import { DeviceMotion } from "expo-sensors";
import { Pedometer } from "expo-sensors";
import * as Location from "expo-location";
import { DeviceSensorPayload } from "./useWebSocket";

// Constants for velocity calculation (from VelocityReadout.tsx)
const DRAG = 0.98;
const STILL_ACCEL_THRESHOLD = 0.15;
const STILL_ROTATION_THRESHOLD = 5;
const MIN_DT = 0.001;
const MAX_DT = 0.1;

const computeAverages = (
  samples: Array<{ t: number; vx: number; vz: number; speed: number }>,
  cutoff: number
): { vx: number; vz: number; speed: number } => {
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

const resolveHeadingDeg = (heading: Location.LocationHeadingObject) => {
  if (typeof heading.trueHeading === "number" && heading.trueHeading >= 0) {
    return heading.trueHeading;
  }
  if (typeof heading.magHeading === "number") {
    return heading.magHeading;
  }
  return null;
};

interface UseDeviceSensorsConfig {
  enabled: boolean;
  onSensorUpdate: (payload: DeviceSensorPayload) => void;
  updateInterval?: number; // ms
}

export function useDeviceSensors({
  enabled,
  onSensorUpdate,
  updateInterval = 100, // 10Hz updates to server
}: UseDeviceSensorsConfig) {
  // State for velocity tracking
  const velocityRef = useRef({ vx: 0, vz: 0 });
  const lastTRef = useRef<number | null>(null);
  const samplesRef = useRef<
    Array<{ t: number; vx: number; vz: number; speed: number }>
  >([]);
  const headingRef = useRef<number | null>(null);
  
  // State for pedometer
  const [stepsSinceOpen, setStepsSinceOpen] = useState(0);
  const stepsHistoryRef = useRef<Array<{ t: number; count: number }>>([]);

  // Pedometer setup
  useEffect(() => {
    if (!enabled) return;

    let subscription: Pedometer.Subscription | null = null;

    const startPedometer = async () => {
      const isAvailable = await Pedometer.isAvailableAsync();
      if (isAvailable) {
        subscription = Pedometer.watchStepCount((result) => {
          setStepsSinceOpen((prev) => prev + result.steps);
          stepsHistoryRef.current.push({ t: Date.now() / 1000, count: result.steps });
        });
      }
    };

    startPedometer();

    return () => {
      subscription?.remove();
    };
  }, [enabled]);

  // Heading setup
  useEffect(() => {
    if (!enabled) return;

    let subscription: Location.LocationSubscription | null = null;
    let isMounted = true;

    const startHeading = async () => {
      try {
        const permission = await Location.requestForegroundPermissionsAsync();
        if (!isMounted) return;
        if (permission.status !== "granted") {
          headingRef.current = null;
          return;
        }

        subscription = await Location.watchHeadingAsync((heading) => {
          headingRef.current = resolveHeadingDeg(heading);
        });
      } catch (error) {
        headingRef.current = null;
      }
    };

    startHeading();

    return () => {
      isMounted = false;
      subscription?.remove();
    };
  }, [enabled]);

  // Device Motion setup & Main Loop
  useEffect(() => {
    if (!enabled) return;

    let motionSubscription: ReturnType<typeof DeviceMotion.addListener> | null = null;
    let throttleTimeout: NodeJS.Timeout | null = null;

    const startMotion = async () => {
      const { status } = await DeviceMotion.requestPermissionsAsync();
      if (status !== "granted") return;

      // Fast sampling for integration (50ms)
      DeviceMotion.setUpdateInterval(50);

      motionSubscription = DeviceMotion.addListener(({ acceleration, rotationRate }) => {
        if (!acceleration) return;

        // --- Velocity Integration Logic ---
        const timestampSeconds = acceleration.timestamp > 1_000_000
            ? acceleration.timestamp / 1000
            : acceleration.timestamp; // Handle different platform units
            
        const lastT = lastTRef.current;
        if (lastT === null) {
          lastTRef.current = timestampSeconds;
          return;
        }

        const dt = Math.min(MAX_DT, Math.max(MIN_DT, timestampSeconds - lastT));
        lastTRef.current = timestampSeconds;

        let { vx, vz } = velocityRef.current;

        // Simple Euler integration
        vx += acceleration.x * dt;
        vz += acceleration.z * dt; // Z is forward/back in device frame usually

        // Apply drag/friction
        vx *= DRAG;
        vz *= DRAG;

        // Zero velocity if device is still
        const aMag = Math.hypot(acceleration.x, acceleration.y, acceleration.z);
        const rotationBeta = rotationRate?.beta ?? 0;
        const rMag = rotationRate
          ? Math.hypot(rotationRate.alpha ?? 0, rotationBeta, rotationRate.gamma ?? 0)
          : 0;
        
        if (aMag < STILL_ACCEL_THRESHOLD && rMag < STILL_ROTATION_THRESHOLD) {
          vx = 0;
          vz = 0;
        }

        velocityRef.current = { vx, vz };
        const speed = Math.hypot(vx, vz);

        // Store sample for averaging
        samplesRef.current.push({ t: timestampSeconds, vx, vz, speed });
        
        // Clean up old samples (> 2 seconds)
        const cutoffHistory = timestampSeconds - 2.0;
        while (samplesRef.current.length > 0 && samplesRef.current[0].t < cutoffHistory) {
          samplesRef.current.shift();
        }

        // --- Throttle Updates to Server ---
        // Only trigger callback every `updateInterval` ms (approx)
        if (!throttleTimeout) {
          throttleTimeout = setTimeout(() => {
            // Calculate averages
            const avg1s = computeAverages(samplesRef.current, timestampSeconds - 1.0);
            
            // Calculate steps in last 3 seconds
            const now = Date.now() / 1000;
            const steps3s = stepsHistoryRef.current
              .filter(s => s.t > now - 3.0)
              .reduce((acc, s) => acc + s.count, 0);
            
            // Clean up old step history
            stepsHistoryRef.current = stepsHistoryRef.current.filter(s => s.t > now - 5.0);

            const payload: DeviceSensorPayload = {
              speed_mps: speed,
              speed_avg_1s_mps: avg1s.speed,
              velocity_x_mps: vx,
              velocity_z_mps: vz,
              heading_deg: headingRef.current,
              steps_last_3s: steps3s,
              steps_since_open: stepsSinceOpen, // We need to access state, might be stale in closure
            };

            onSensorUpdate(payload);
            throttleTimeout = null;
          }, updateInterval);
        }
      });
    };

    startMotion();

    return () => {
      motionSubscription?.remove();
      if (throttleTimeout) clearTimeout(throttleTimeout);
    };
  }, [enabled, onSensorUpdate, updateInterval, stepsSinceOpen]); // dependency on stepsSinceOpen updates closure
}
