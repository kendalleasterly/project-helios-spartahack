import { useEffect } from "react";
import { Magnetometer, MagnetometerMeasurement } from "expo-sensors";

type MagnetometerSubscription = ReturnType<typeof Magnetometer.addListener>;

export function MagnetometerLogger() {
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

      // 200ms keeps us within Android 12+ default sensor rate limits.
      Magnetometer.setUpdateInterval(200);

      subscription = Magnetometer.addListener(
        (measurement: MagnetometerMeasurement) => {
          console.log("[magnetometer]", measurement);
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
