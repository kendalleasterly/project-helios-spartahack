import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as Haptics from "expo-haptics";

type HapticsButton = {
  key: string;
  label: string;
  onPress: () => void;
};

const runBurst = () => {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  }, 120);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  }, 240);
};

const runLongBurst = () => {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  }, 140);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid);
  }, 280);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  }, 420);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid);
  }, 560);
  setTimeout(() => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
  }, 700);
};

const BUTTONS: HapticsButton[] = [
  {
    key: "heavy",
    label: "Heavy Impact",
    onPress: () => void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy),
  },
  {
    key: "rigid",
    label: "Rigid Impact",
    onPress: () => void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid),
  },
  {
    key: "soft",
    label: "Soft Impact",
    onPress: () => void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Soft),
  },
  {
    key: "error",
    label: "Error Notification",
    onPress: () => void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error),
  },
  {
    key: "warning",
    label: "Warning Notification",
    onPress: () =>
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning),
  },
  {
    key: "success",
    label: "Success Notification",
    onPress: () =>
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success),
  },
  {
    key: "selection",
    label: "Selection Tick",
    onPress: () => void Haptics.selectionAsync(),
  },
  {
    key: "burst",
    label: "Heavy Burst",
    onPress: runBurst,
  },
  {
    key: "long-burst",
    label: "Long Burst",
    onPress: runLongBurst,
  },
];

export function HapticsDemo() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Haptics Demo</Text>
      <View style={styles.grid}>
        {BUTTONS.map((button) => (
          <TouchableOpacity
            key={button.key}
            onPress={button.onPress}
            style={styles.button}
            activeOpacity={0.8}
          >
            <Text style={styles.buttonText}>{button.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignSelf: "stretch",
    width: "100%",
    backgroundColor: "rgba(15, 23, 42, 0.85)",
    borderRadius: 16,
    padding: 14,
    gap: 12,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
  },
  title: {
    fontSize: 12,
    fontWeight: "700",
    color: "#E2E8F0",
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  button: {
    flexBasis: "48%",
    flexGrow: 1,
    backgroundColor: "rgba(226, 232, 240, 0.1)",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.3)",
  },
  buttonText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#F8FAFC",
    textAlign: "center",
  },
});
