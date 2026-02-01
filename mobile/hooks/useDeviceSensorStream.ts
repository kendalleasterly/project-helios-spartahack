import { useEffect, useRef, useState } from "react";
import { DeviceMotion, Pedometer } from "expo-sensors";
import * as Location from "expo-location";
import type { DeviceSensorPayload } from "./useWebSocket";

type UseDeviceSensorStreamOptions = {
  enabled: boolean;
  updateIntervalMs?: number;
};

type VelocityAverages = {
  vx: number;
  vz: number;
  speed: number;
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

const sanitizeValue = (value: number | null | undefined): number =>
  Number.isFinite(value) ? (value as number) : 0;

const resolveHeadingDeg = (heading: Location.LocationHeadingObject) => {
  if (typeof heading.trueHeading === "number" && heading.trueHeading >= 0) {
    return heading.trueHeading;
  }
  if (typeof heading.magHeading === "number") {
    return heading.magHeading;
  }
  return null;
};

export const useDeviceSensorStream = ({
  enabled,
  updateIntervalMs = 50,
}: UseDeviceSensorStreamOptions) => {
  const velocityRef = useRef({ dx: 0, dz: 0 });
  const lastTRef = useRef<number | null>(null);
  const rotationBetaRef = useRef(0);
  const stepsSinceOpenRef = useRef(0);
  const lastStepsRef = useRef<number | null>(null);
  const stepsWindowRef = useRef<Array<{ t: number; count: number }>>([]);
  const samplesRef = useRef<
    Array<{ t: number; vx: number; vz: number; speed: number }>
  >([]);
  const speedAvg1sRef = useRef(0);
  const headingRef = useRef<number | null>(null);
  const [speedMps, setSpeedMps] = useState(0);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let motionSubscription: ReturnType<typeof DeviceMotion.addListener> | null =
      null;
    let pedometerSubscription: { remove: () => void } | null = null;
    let headingSubscription: Location.LocationSubscription | null = null;
    let isMounted = true;

    const start = async () => {
      const motionAvailable = await DeviceMotion.isAvailableAsync();
      if (!motionAvailable) {
        console.warn("Device motion not available on this device.");
        return;
      }

      const motionPermission = await DeviceMotion.getPermissionsAsync();
      if (!motionPermission.granted) {
        const request = await DeviceMotion.requestPermissionsAsync();
        if (!request.granted) {
          console.warn("Device motion permission not granted.");
          return;
        }
      }

      if (!isMounted) {
        return;
      }

      DeviceMotion.setUpdateInterval(updateIntervalMs);

      const pedometerAvailable = await Pedometer.isAvailableAsync();
      if (!pedometerAvailable) {
        console.warn("Pedometer not available on this device.");
      } else {
        const pedometerPermission = await Pedometer.requestPermissionsAsync();
        if (!pedometerPermission.granted) {
          console.warn("Pedometer permission not granted.");
        } else {
          pedometerSubscription = Pedometer.watchStepCount(({ steps }) => {
            const now = Date.now();
            const lastSteps = lastStepsRef.current;

            if (lastSteps === null) {
              lastStepsRef.current = steps;
              stepsSinceOpenRef.current = steps;
              return;
            }

            const delta = steps - lastSteps;
            lastStepsRef.current = steps;
            stepsSinceOpenRef.current = steps;

            if (delta > 0) {
              stepsWindowRef.current.push({ t: now, count: delta });
            } else if (delta < 0) {
              stepsWindowRef.current = [];
            }

            const cutoff = now - 3000;
            while (
              stepsWindowRef.current.length > 0 &&
              stepsWindowRef.current[0].t < cutoff
            ) {
              stepsWindowRef.current.shift();
            }
          });
        }
      }

      try {
        const permission = await Location.requestForegroundPermissionsAsync();
        if (!isMounted) {
          return;
        }
        if (permission.status === "granted") {
          headingSubscription = await Location.watchHeadingAsync((heading) => {
            headingRef.current = resolveHeadingDeg(heading);
          });
        }
      } catch (error) {
        headingRef.current = null;
      }

      motionSubscription = DeviceMotion.addListener(
        ({ acceleration, rotationRate }) => {
          if (!acceleration) {
            return;
          }

          const nowSeconds = Date.now() / 1000;
          const rawTimestamp =
            typeof acceleration.timestamp === "number" &&
            Number.isFinite(acceleration.timestamp) &&
            acceleration.timestamp > 0
              ? acceleration.timestamp
              : nowSeconds;
          const timestampSeconds =
            rawTimestamp > 1_000_000 ? rawTimestamp / 1000 : rawTimestamp;

          const lastT = lastTRef.current;
          if (lastT === null) {
            lastTRef.current = timestampSeconds;
            return;
          }

          const safeTimestamp =
            timestampSeconds <= lastT ? nowSeconds : timestampSeconds;
          const dt = Math.min(
            MAX_DT,
            Math.max(MIN_DT, safeTimestamp - lastT)
          );
          lastTRef.current = safeTimestamp;

          let { dx, dz } = velocityRef.current;

          dx += sanitizeValue(acceleration.x) * dt;
          dz += sanitizeValue(acceleration.z) * dt;

          dx *= DRAG;
          dz *= DRAG;

          const aMag = Math.hypot(
            sanitizeValue(acceleration.x),
            sanitizeValue(acceleration.y),
            sanitizeValue(acceleration.z)
          );
          const rotationBeta = sanitizeValue(rotationRate?.beta);
          rotationBetaRef.current = rotationBeta;
          const rMag = rotationRate
            ? Math.hypot(
                sanitizeValue(rotationRate.alpha),
                rotationBeta,
                sanitizeValue(rotationRate.gamma)
              )
            : 0;
          const isStill =
            aMag < STILL_ACCEL_THRESHOLD && rMag < STILL_ROTATION_THRESHOLD;

          if (isStill) {
            dx = 0;
            dz = 0;
          }

          velocityRef.current = { dx, dz };

          const speed = Math.hypot(dx, dz);
          const samples = samplesRef.current;
          const sampleTime = safeTimestamp;
          samples.push({ t: sampleTime, vx: dx, vz: dz, speed });

          const cutoff1000 = sampleTime - 1.0;
          while (samples.length > 0 && samples[0].t < cutoff1000) {
            samples.shift();
          }

          const avg1s = computeAverages(samples, cutoff1000);
          speedAvg1sRef.current = avg1s.speed;
          setSpeedMps(avg1s.speed);
        }
      );
    };

    start();

    return () => {
      isMounted = false;
      if (motionSubscription) {
        motionSubscription.remove();
      }
      if (pedometerSubscription) {
        pedometerSubscription.remove();
      }
      if (headingSubscription) {
        headingSubscription.remove();
      }
    };
  }, [enabled, updateIntervalMs]);

  const getSnapshot = (): DeviceSensorPayload => {
    const now = Date.now();
    const window = stepsWindowRef.current;
    const cutoff = now - 3000;
    while (window.length > 0 && window[0].t < cutoff) {
      window.shift();
    }
    const stepsLast3s = window.reduce((sum, entry) => sum + entry.count, 0);
    const { dx, dz } = velocityRef.current;

    const speedInstant = Math.hypot(dx, dz);

    return {
      speed_mps: speedInstant,
      speed_avg_1s_mps: speedAvg1sRef.current,
      velocity_x_mps: dx,
      velocity_z_mps: dz,
      heading_deg: headingRef.current,
      steps_last_3s: stepsLast3s,
      steps_since_open: stepsSinceOpenRef.current,
    };
  };

  return { getSnapshot, speedMps };
};
