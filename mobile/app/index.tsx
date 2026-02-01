import { MagnetometerReadout } from "@/components/MagnetometerReadout"
import { VelocityReadout } from "@/components/VelocityReadout"
import { registerRootComponent } from "expo"
import { View } from "react-native"
import { SafeAreaProvider } from "react-native-safe-area-context"

function App() {
	return (
		<SafeAreaProvider>
			{/* <AudioStreamTest /> */}
			{/* <AccelerometerLogger /> */}
			{/* <MagnetometerLogger /> */}
			<View>
				<VelocityReadout />
				<MagnetometerReadout />
			</View>
		</SafeAreaProvider>
	)
}

registerRootComponent(App)
