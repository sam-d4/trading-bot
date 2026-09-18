import { useEffect, useRef, useState } from 'react'

export interface LiveEvent {
  type: string
  payload: Record<string, unknown>
  at: string
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected'

/** Connects to the backend's /ws/live event relay and calls onEvent for every message.
 * Reconnects automatically with a short fixed backoff on disconnect - this is a long-lived
 * dashboard tab, not a one-shot request, so it should keep trying rather than give up.
 * Returns the current connection state so the UI can show a live/reconnecting indicator. */
export function useLiveEvents(onEvent: (event: LiveEvent) => void): ConnectionState {
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent
  const [state, setState] = useState<ConnectionState>('connecting')

  useEffect(() => {
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let stopped = false

    function connect() {
      setState('connecting')
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/live`)

      socket.onopen = () => setState('connected')

      socket.onmessage = (msg) => {
        try {
          const event = JSON.parse(msg.data) as LiveEvent
          onEventRef.current(event)
        } catch {
          // ignore malformed frames
        }
      }

      socket.onclose = () => {
        setState('disconnected')
        if (!stopped) reconnectTimer = setTimeout(connect, 2000)
      }

      socket.onerror = () => {
        socket?.close()
      }
    }

    connect()

    return () => {
      stopped = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  return state
}
