import { AudioStreamTest } from "@/components/AudioStreamTest"
import { MagnetometerReadout } from "@/components/MagnetometerReadout"
import { registerRootComponent } from "expo"
import { SafeAreaProvider } from "react-native-safe-area-context"

function App() {
	return (
		<SafeAreaProvider>
			<AudioStreamTest />
			{/* <AccelerometerLogger />
			<VelocityReadout /> */}
			<MagnetometerReadout />
		</SafeAreaProvider>
	)
}

registerRootComponent(App)
