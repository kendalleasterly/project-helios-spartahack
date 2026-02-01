Yep — Deepgram has an official “go raw” path: open a plain WebSocket to **`wss://api.deepgram.com/v1/listen`**, pass the right **query params** (especially `encoding` + `sample_rate` for raw PCM), then **send binary audio frames** and read JSON transcripts. ([Deepgram Docs][1])

Below is a **concrete, no-backend, Expo RN** example you can copy/paste and adapt.

---

## What Deepgram expects in “raw WebSocket” mode

### 1) WebSocket URL

Base endpoint is:

- `wss://api.deepgram.com/v1/listen` ([Deepgram Docs][1])

### 2) Authentication (no backend)

Deepgram’s reference says you can authenticate either with an `Authorization` header **or** by passing a token via the `token` query parameter. ([Deepgram Docs][1])

For “no backend”, the **`token=` query param** is usually simplest in React Native.

### 3) Raw audio requires `encoding` + `sample_rate`

Deepgram explicitly states: when you send **raw, headerless** audio packets, `encoding` is required — and if you use `encoding`, then `sample_rate` is also required. ([Deepgram Docs][2])

For Expo PCM 16-bit mono @ 16kHz, you’ll typically use:

- `encoding=linear16`
- `sample_rate=16000`
- `channels=1`

### 4) Low latency knobs

- `interim_results=true` for partial text as the user speaks. ([Deepgram Docs][3])
- `endpointing=300` (ms) to finalize after short pauses (tune 200–700). ([Deepgram Docs][4])

### 5) KeepAlive (avoid NET-0001)

If there are pauses / silence, Deepgram recommends sending a `KeepAlive` **text** message every 3–5 seconds; otherwise you can hit a ~10s timeout and the connection closes (NET-0001). ([Deepgram Docs][5])

### 6) Stop cleanly

- `Finalize` flushes buffered audio into final transcript. ([Deepgram Docs][6])
- `CloseStream` closes the stream and returns final transcription + metadata. ([Deepgram Docs][7])

---

## Concrete Expo RN implementation (raw WebSocket)

This is written to work with an Expo “audio stream” module that calls `onAudioStream({ data: <base64> })` and can be configured for PCM 16-bit / 16kHz / mono. If your module’s function names differ, keep the **shape** the same.

### Install

```bash
npm i base64-js
```

### `deepgramRawWs.ts`

```ts
import { toByteArray } from "base64-js"

// ---- 1) Build the Deepgram streaming URL ----
// Deepgram's live streaming endpoint is wss://api.deepgram.com/v1/listen.
// Auth can be via Authorization header OR token query param. (Docs show token query param option.)
export function makeDeepgramUrl(params: {
	apiKeyOrTempToken: string // hackathon mode: API key is fine; otherwise use a temp token
	model?: string // e.g. "nova-3"
	sampleRateHz: number // MUST match the mic stream output
	channels: 1 | 2
	interimResults?: boolean
	endpointingMs?: number
}) {
	const qs = new URLSearchParams()

	qs.set("model", params.model ?? "nova-3")

	// RAW PCM requires encoding + sample_rate when sending headerless audio.
	qs.set("encoding", "linear16") // PCM16
	qs.set("sample_rate", String(params.sampleRateHz))
	qs.set("channels", String(params.channels))

	// Low-latency UX
	qs.set("interim_results", params.interimResults === false ? "false" : "true")
	if (params.endpointingMs != null)
		qs.set("endpointing", String(params.endpointingMs))

	// Convenient no-backend auth: token query param
	qs.set("token", params.apiKeyOrTempToken)

	return `wss://api.deepgram.com/v1/listen?${qs.toString()}`
}

type DeepgramResult = {
	type?: string // often "Results" (see response schema)
	is_final?: boolean
	speech_final?: boolean
	channel?: {
		alternatives?: Array<{ transcript?: string; confidence?: number }>
	}
	metadata?: any
	// ...lots more fields possible
}

type TranscriptHandler = (msg: {
	text: string
	isFinal: boolean
	speechFinal: boolean
	raw: DeepgramResult
}) => void

export class DeepgramRawWsClient {
	private ws: WebSocket | null = null
	private keepAliveTimer: any | null = null

	constructor(private onTranscript: TranscriptHandler) {}

	connect(url: string) {
		// NOTE: You can also pass headers for Authorization in React Native:
		// new WebSocket(url, undefined, { headers: { Authorization: `Token ${KEY}` } })
		// But token query param is simplest for no-backend.
		this.ws = new WebSocket(url)

		// Helps some environments interpret incoming binary; usually optional in RN
		// (In browsers you'd do ws.binaryType = "arraybuffer"; RN support varies.)
		// (this.ws as any).binaryType = "arraybuffer";

		this.ws.onopen = () => {
			// KeepAlive: send as TEXT frames every 3–5 seconds during silence.
			// Deepgram warns KeepAlive must be a text WebSocket frame.
			this.keepAliveTimer = setInterval(() => {
				if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
				this.ws.send(JSON.stringify({ type: "KeepAlive" })) // TEXT frame
			}, 4000)
		}

		this.ws.onmessage = (evt) => {
			// Deepgram transcript messages are JSON per the response schema
			// (contains channel.alternatives[0].transcript, is_final, speech_final, etc.)
			try {
				const data = JSON.parse(String(evt.data)) as DeepgramResult
				const text = data?.channel?.alternatives?.[0]?.transcript ?? ""
				if (!text) return

				this.onTranscript({
					text,
					isFinal: Boolean(data.is_final),
					speechFinal: Boolean(data.speech_final),
					raw: data,
				})
			} catch {
				// ignore non-JSON
			}
		}

		this.ws.onerror = (e) => {
			// Most useful debugging: ensure audio actually arrives within ~10s,
			// and encoding/sample_rate match your stream.
			console.warn("Deepgram WS error:", e)
		}

		this.ws.onclose = () => {
			this.cleanup()
		}
	}

	/**
	 * Send one chunk of mic audio to Deepgram.
	 * IMPORTANT: send BINARY audio, not base64 strings.
	 */
	sendAudioBase64Chunk(b64: string) {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return

		const u8 = toByteArray(b64) // Uint8Array of PCM bytes

		// RN WebSocket can send ArrayBuffer. We must slice to exact byte range.
		const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)
		this.ws.send(ab) // BINARY frame containing raw PCM16 bytes
	}

	/**
	 * Clean stop:
	 * - Finalize flushes buffered audio to final results
	 * - CloseStream closes the connection and returns final + metadata
	 * Both are TEXT JSON control messages.
	 */
	stop() {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return

		this.ws.send(JSON.stringify({ type: "Finalize" }))
		this.ws.send(JSON.stringify({ type: "CloseStream" }))
		// Deepgram will close from server side after sending remaining messages.
	}

	closeImmediately() {
		this.ws?.close()
		this.cleanup()
	}

	private cleanup() {
		if (this.keepAliveTimer) clearInterval(this.keepAliveTimer)
		this.keepAliveTimer = null
		this.ws = null
	}
}
```

### Hook it up to your Expo audio stream module

The only missing piece is the function that starts mic streaming and calls you back with base64 PCM chunks. Here’s the pattern:

```ts
// exampleUsage.ts
import { DeepgramRawWsClient, makeDeepgramUrl } from "./deepgramRawWs"

// Replace with your own import / calls from your Expo streaming module
// (must be configured to output PCM16, 16kHz, mono)
import { startRecording, stopRecording } from "YOUR_EXPO_AUDIO_STREAM_MODULE"

const dg = new DeepgramRawWsClient(({ text, isFinal }) => {
	console.log(isFinal ? "[final]" : "[partial]", text)
})

export async function startLiveStt() {
	const url = makeDeepgramUrl({
		apiKeyOrTempToken: process.env.EXPO_PUBLIC_DEEPGRAM_KEY!,
		sampleRateHz: 16000,
		channels: 1,
		interimResults: true,
		endpointingMs: 300,
	})

	dg.connect(url)

	await startRecording({
		sampleRate: 16000,
		channels: 1,
		encoding: "pcm_16bit",
		interval: 80, // smaller = lower latency (more overhead)
		onAudioStream: (evt: { data: string }) => {
			dg.sendAudioBase64Chunk(evt.data)
		},
	})
}

export async function stopLiveStt() {
	await stopRecording()
	dg.stop() // Finalize + CloseStream
}
```

---

## If you got a Deepgram error, it’s usually one of these

### Connection closes ~10 seconds in (often NET-0001)

Cause: no audio (or no KeepAlive) is arriving during silence. Deepgram recommends KeepAlive every 3–5 seconds; otherwise it times out and closes. ([Deepgram Docs][5])

### Immediate disconnect

Deepgram’s lower-level guide lists common reasons:

- no audio making it through
- encoding not supported or `encoding` param doesn’t match the actual stream
- invalid query params ([Deepgram Docs][8])

### “No transcript” / garbage transcript

Almost always mismatch between:

- what you _think_ you’re sending (PCM16 16kHz mono) and what you’re _actually_ sending
  Deepgram explicitly stresses understanding whether your audio is raw vs containerized, and matching params accordingly. ([Deepgram Docs][2])

---

## Quick check I’d do next (takes 30 seconds)

Print the first chunk size and ensure it’s non-trivial:

- 80ms of 16kHz mono PCM16 is:
  16,000 samples/sec × 0.08 sec × 2 bytes/sample ≈ **2,560 bytes** per chunk

So if your decoded `Uint8Array` chunks are like 10 bytes or 100 bytes, something’s off in capture.

---

If you paste the exact Deepgram error payload you saw (or the close code / reason), I’ll map it to the fix in the docs and point to which part of the pipeline (auth vs audio format vs keepalive vs params) is failing.

[1]: https://developers.deepgram.com/reference/speech-to-text/listen-streaming "Live Audio | Deepgram's Docs"
[2]: https://developers.deepgram.com/docs/encoding "Encoding | Deepgram's Docs"
[3]: https://developers.deepgram.com/docs/interim-results?utm_source=chatgpt.com "Interim Results | Deepgram's Docs"
[4]: https://developers.deepgram.com/docs/understand-endpointing-interim-results?utm_source=chatgpt.com "Configure Endpointing and Interim Results"
[5]: https://developers.deepgram.com/docs/audio-keep-alive "Audio Keep Alive | Deepgram's Docs"
[6]: https://developers.deepgram.com/docs/finalize "Finalize | Deepgram's Docs"
[7]: https://developers.deepgram.com/docs/close-stream "Close Stream | Deepgram's Docs"
[8]: https://developers.deepgram.com/docs/lower-level-websockets "Using Lower-Level Websockets with the Streaming API | Deepgram's Docs"
