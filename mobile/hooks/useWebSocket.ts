import { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { BACKEND_SERVER_URL } from "@env";

const SERVER_URL = BACKEND_SERVER_URL || "http://192.168.137.1:8000";

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

interface UseWebSocketReturn {
  socket: Socket | null;
  status: ConnectionStatus;
  sendFrame: (base64Frame: string, debug?: boolean) => void;
  connect: () => void;
  disconnect: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const socketRef = useRef<Socket | null>(null);

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

    socket.on("connect_error", (error: Error) => {
      console.error("WebSocket connection error:", error.message);
      setStatus("error");
    });

    socket.on("error", (error: unknown) => {
      console.error("WebSocket error:", error);
      setStatus("error");
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

  const sendFrame = useCallback((base64Frame: string, debug = false) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("video_frame", {
        frame: base64Frame,
        debug,
      });
    } else {
      console.warn("Cannot send frame: WebSocket not connected");
    }
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
    connect,
    disconnect,
  };
}
