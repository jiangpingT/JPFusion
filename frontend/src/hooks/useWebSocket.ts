/**
 * useWebSocket.ts — WebSocket 封装 hook
 *
 * 功能：
 *   - 自动连接 WebSocket
 *   - 断线自动重连（最多 5 次，指数退避）
 *   - 将收到的 JSON 消息传给回调
 *   - 组件卸载时自动关闭连接
 */

import { useEffect, useRef, useCallback } from 'react';

type MessageCallback = (data: Record<string, unknown>) => void;

interface UseWebSocketOptions {
  url: string;
  onMessage: MessageCallback;
  onOpen?: () => void;
  onClose?: () => void;
  maxRetries?: number;
}

export function useWebSocket({
  url,
  onMessage,
  onOpen,
  onClose,
  maxRetries = 5,
}: UseWebSocketOptions) {
  const wsRef      = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        onOpen?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch {
          // 忽略非 JSON 消息
        }
      };

      ws.onclose = () => {
        onClose?.();
        if (mountedRef.current && retriesRef.current < maxRetries) {
          const delay = Math.min(1000 * 2 ** retriesRef.current, 10000);
          retriesRef.current += 1;
          setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // 连接失败，等待重连
    }
  }, [url, onMessage, onOpen, onClose, maxRetries]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [connect]);
}
