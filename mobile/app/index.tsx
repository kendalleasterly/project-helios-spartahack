import { AudioStreamTest } from "@/components/AudioStreamTest"
import { registerRootComponent } from "expo"
import { SafeAreaProvider } from "react-native-safe-area-context"

function App() {
	return (
		<SafeAreaProvider>
			<AudioStreamTest />
			{/* <AccelerometerLogger />
			<MagnetometerLogger />
			<View>
				<VelocityReadout />
				<MagnetometerReadout />
			</View> */}
		</SafeAreaProvider>
	)
}

registerRootComponent(App)
