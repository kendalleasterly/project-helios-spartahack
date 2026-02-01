import { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { BACKEND_SERVER_URL } from "@env";

const SERVER_URL = BACKEND_SERVER_URL || "https://impolite-sky-noncontemplatively.ngrok-free.dev";
console.log('WebSocket connecting to:', SERVER_URL);

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

export interface TextTokenEvent {
  token: string;
  emergency: boolean;
  is_first: boolean;
}

// Backend main branch sends text_response events with this format
interface TextResponseEvent {
  text: string;
  mode: "vision" | "conversation";
  emergency: boolean;
}

export type TextTokenCallback = (event: TextTokenEvent) => void;

interface UseWebSocketReturn {
  socket: Socket | null;
  status: ConnectionStatus;
  sendFrame: (base64Frame: string, userQuestion?: string, debug?: boolean) => void;
  onTextToken: (callback: TextTokenCallback) => void;
  connect: () => void;
  disconnect: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const socketRef = useRef<Socket | null>(null);
  const textTokenCallbackRef = useRef<TextTokenCallback | null>(null);

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    setStatus("connecting");

    const socket = io(SERVER_URL, {
      transports: ["websocket"],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
    });

    socket.on("connect", () => {
      console.log("WebSocket connected");
      setStatus("connected");
    });

    socket.on("connection_established", (data: unknown) => {
      console.log("Connection established:", data);
    });

    socket.on("disconnect", (reason: string) => {
      console.log("WebSocket disconnected:", reason);
      setStatus("disconnected");
    });

    socket.on("reconnect_attempt", (attemptNumber: number) => {
      console.log(`WebSocket reconnect attempt #${attemptNumber}`);
      setStatus("connecting");
    });

    socket.on("connect_error", () => {
      setStatus("connecting");
    });

    socket.on("error", () => {
      setStatus("connecting");
    });

    // Listen for streaming text_token events (feature/audio-tts branch backend)
    socket.on("text_token", (data: TextTokenEvent) => {
      console.log('[WebSocket] text_token received:', JSON.stringify(data));
      if (textTokenCallbackRef.current) {
        textTokenCallbackRef.current(data);
      } else {
        console.warn('[WebSocket] text_token received but no callback registered');
      }
    });

    // Listen for complete text_response events (main branch backend)
    // Convert to TextTokenEvent format for TTS pipeline compatibility
    socket.on("text_response", (data: TextResponseEvent) => {
      console.log('[WebSocket] text_response received:', JSON.stringify(data));
      if (textTokenCallbackRef.current && data.text) {
        // Send the complete text as a single token with sentence terminator
        // This ensures the text buffer will flush and trigger TTS
        const textWithPunctuation = data.text.endsWith('.') || 
                                    data.text.endsWith('!') || 
                                    data.text.endsWith('?') 
          ? data.text 
          : data.text + '.';
        
        textTokenCallbackRef.current({
          token: textWithPunctuation,
          emergency: data.emergency,
          is_first: true,
        });
        console.log('[WebSocket] Converted text_response to text_token for TTS');
      } else if (!textTokenCallbackRef.current) {
        console.warn('[WebSocket] text_response received but no callback registered');
      }
    });

    // Log all events for debugging - helps identify what backend is sending
    socket.onAny((eventName: string, ...args: unknown[]) => {
      if (eventName !== 'text_token' && eventName !== 'text_response' && 
          eventName !== 'connect' && eventName !== 'disconnect') {
        console.log(`[WebSocket] Event "${eventName}":`, JSON.stringify(args).slice(0, 200));
      }
    });

    socketRef.current = socket;
  }, []);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setStatus("disconnected");
    }
  }, []);

  const sendFrame = useCallback((base64Frame: string, userQuestion?: string, debug = false) => {
    if (socketRef.current?.connected) {
      if (userQuestion) {
        console.log(`Sending frame with question to backend: "${userQuestion}"`);
      } else {
        console.log(`Sending frame to backend (${base64Frame.length} bytes)`);
      }
      socketRef.current.emit("video_frame_streaming", {
        frame: base64Frame,
        user_question: userQuestion,
        debug,
      });
    } else {
      console.warn('Cannot send frame: socket not connected');
    }
  }, []);

  const onTextToken = useCallback((callback: TextTokenCallback) => {
    textTokenCallbackRef.current = callback;
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    socket: socketRef.current,
    status,
    sendFrame,
    onTextToken,
    connect,
    disconnect,
  };
}
