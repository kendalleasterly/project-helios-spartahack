import { useEffect } from "react";
import { Accelerometer, AccelerometerMeasurement } from "expo-sensors";

type AccelerometerSubscription = ReturnType<typeof Accelerometer.addListener>;

export function AccelerometerLogger() {
  useEffect(() => {
    let subscription: AccelerometerSubscription | null = null;
    let isMounted = true;

    const start = async () => {
      const isAvailable = await Accelerometer.isAvailableAsync();
      if (!isAvailable) {
        console.warn("Accelerometer not available on this device.");
        return;
      }

      const permission = await Accelerometer.getPermissionsAsync();
      if (!permission.granted) {
        const request = await Accelerometer.requestPermissionsAsync();
        if (!request.granted) {
          console.warn("Accelerometer permission not granted.");
          return;
        }
      }

      if (!isMounted) {
        return;
      }

      // 200ms keeps us within Android 12+ default sensor rate limits.
      Accelerometer.setUpdateInterval(200);

      subscription = Accelerometer.addListener(
        (measurement: AccelerometerMeasurement) => {
          console.log("[accelerometer]", measurement);
        }
      );
    };

    start();

    return () => {
      isMounted = false;
      if (subscription) {
        subscription.remove();
      }
    };
  }, []);

  return null;
}
