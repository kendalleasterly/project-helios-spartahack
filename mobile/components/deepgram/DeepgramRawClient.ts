export type DeepgramAuthMode = "header" | "token";

export type DeepgramConfig = {
  apiKey: string;
  sampleRate: number;
  channels: 1;
  model: string;
  language: string;
  interimResults: boolean;
  endpointingMs: number;
  utteranceEndMs: number;
  vadEvents: boolean;
  punctuate: boolean;
  smartFormat: boolean;
  noDelay: boolean;
  keepAliveMs: number;
  enableKeepAlive: boolean;
  authMode: DeepgramAuthMode;
};

export type DeepgramEvent =
  | { type: "open" }
  | { type: "close"; code: number; reason: string }
  | { type: "error"; message: string }
  | { type: "message"; raw: string; parsed: unknown | null };

export type DeepgramClientHandlers = {
  onEvent: (event: DeepgramEvent) => void;
};

type WebSocketConstructorWithHeaders = new (
  url: string,
  protocols?: string | string[],
  options?: { headers?: Record<string, string> },
) => WebSocket;

const buildDeepgramUrl = (config: DeepgramConfig): string => {
  const params = new URLSearchParams({
    model: config.model,
    language: config.language,
    encoding: "linear16",
    sample_rate: String(config.sampleRate),
    channels: String(config.channels),
    interim_results: config.interimResults ? "true" : "false",
    punctuate: config.punctuate ? "true" : "false",
    smart_format: config.smartFormat ? "true" : "false",
    no_delay: config.noDelay ? "true" : "false",
    endpointing: String(config.endpointingMs),
    utterance_end_ms: String(config.utteranceEndMs),
    vad_events: config.vadEvents ? "true" : "false",
  });

  if (config.authMode === "token") {
    params.set("token", config.apiKey);
  }

  return `wss://api.deepgram.com/v1/listen?${params.toString()}`;
};

const createDeepgramSocket = (
  url: string,
  apiKey: string,
  authMode: DeepgramAuthMode,
): WebSocket => {
  if (authMode === "token") {
    return new WebSocket(url);
  }

  const WebSocketWithHeaders = WebSocket as unknown as WebSocketConstructorWithHeaders;
  return new WebSocketWithHeaders(url, undefined, {
    headers: { Authorization: `Token ${apiKey}` },
  });
};

const describeError = (value: unknown): string => {
  if (value instanceof Error) {
    return value.message;
  }

  if (value && typeof value === "object" && "message" in value) {
    const message = (value as { message: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }

  if (typeof value === "string") {
    return value;
  }

  return "Deepgram connection error.";
};

export class DeepgramRawClient {
  private ws: WebSocket | null = null;
  private keepAliveTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private config: DeepgramConfig,
    private handlers: DeepgramClientHandlers,
  ) {}

  async connect(): Promise<void> {
    const url = buildDeepgramUrl(this.config);
    const connection = createDeepgramSocket(
      url,
      this.config.apiKey,
      this.config.authMode,
    );

    this.ws = connection;

    connection.onmessage = (event) => {
      if (typeof event.data !== "string") {
        return;
      }

      let parsed: unknown | null = null;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        parsed = null;
      }

      this.handlers.onEvent({
        type: "message",
        raw: event.data,
        parsed,
      });
    };

    connection.onclose = (event) => {
      this.cleanup();
      this.handlers.onEvent({
        type: "close",
        code: event.code,
        reason: event.reason,
      });
    };

    return new Promise((resolve, reject) => {
      let didOpen = false;

      connection.onopen = () => {
        didOpen = true;
        if (this.config.enableKeepAlive) {
          this.startKeepAlive();
        }
        this.handlers.onEvent({ type: "open" });
        resolve();
      };

      connection.onerror = (error) => {
        const message = describeError(error);
        this.handlers.onEvent({ type: "error", message });
        if (!didOpen) {
          reject(new Error(message));
        }
      };
    });
  }

  sendAudio(buffer: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      this.ws.send(buffer);
    } catch (error) {
      const message = describeError(error);
      this.handlers.onEvent({ type: "error", message });
    }
  }

  finalizeAndClose(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      this.ws.send(JSON.stringify({ type: "Finalize" }));
      this.ws.send(JSON.stringify({ type: "CloseStream" }));
    } catch (error) {
      const message = describeError(error);
      this.handlers.onEvent({ type: "error", message });
    }
  }

  close(): void {
    try {
      this.ws?.close();
    } catch {
      // Ignore close failures.
    }
    this.cleanup();
  }

  isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private startKeepAlive(): void {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
    }

    this.keepAliveTimer = setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return;
      }

      try {
        this.ws.send(JSON.stringify({ type: "KeepAlive" }));
      } catch {}
    }, this.config.keepAliveMs);
  }

  private cleanup(): void {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = null;
    }
    this.ws = null;
  }
}
