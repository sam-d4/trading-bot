import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { api, type EquityPoint, type ModelVersionSummary, type PendingPromotion, type Status, type Trade } from './api'
import { EquityChart } from './components/EquityChart'
import { PerformancePanel } from './components/PerformancePanel'
import { StatusBanner } from './components/StatusBanner'
import { TradesPanel } from './components/TradesPanel'
import { TrainingPanel } from './components/TrainingPanel'
import { useLiveEvents, type LiveEvent } from './hooks/useWebSocket'

function useClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [positions, setPositions] = useState<Trade[]>([])
  const [equity, setEquity] = useState<EquityPoint[]>([])
  const [equityHours, setEquityHours] = useState(24 * 7)
  const [models, setModels] = useState<ModelVersionSummary[]>([])
  const [pending, setPending] = useState<PendingPromotion[]>([])
  const [resetting, setResetting] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [busyModelId, setBusyModelId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const clock = useClock()

  const refreshAll = useCallback(() => {
    api.status().then(setStatus).catch((e) => setError(String(e)))
    api.trades().then(setTrades).catch(() => {})
    api.positions().then(setPositions).catch(() => {})
    api.equity(equityHours).then(setEquity).catch(() => {})
    api.models().then(setModels).catch(() => {})
    api.pendingPromotions().then(setPending).catch(() => {})
  }, [equityHours])

  useEffect(() => {
    refreshAll()
    const interval = setInterval(refreshAll, 30_000) // REST poll fallback if the websocket drops
    return () => clearInterval(interval)
  }, [refreshAll])

  const handleLiveEvent = useCallback(
    (event: LiveEvent) => {
      switch (event.type) {
        case 'trade_opened':
        case 'trade_closed':
          api.trades().then(setTrades).catch(() => {})
          api.positions().then(setPositions).catch(() => {})
          break
        case 'equity_update':
          api.equity(equityHours).then(setEquity).catch(() => {})
          api.status().then(setStatus).catch(() => {})
          break
        case 'kill_switch_tripped':
        case 'kill_switch_reset':
          api.status().then(setStatus).catch(() => {})
          break
        case 'model_promotion_pending':
        case 'model_promoted':
        case 'training_status':
          api.models().then(setModels).catch(() => {})
          api.pendingPromotions().then(setPending).catch(() => {})
          api.status().then(setStatus).catch(() => {})
          break
      }
    },
    [equityHours],
  )

  const connectionState = useLiveEvents(handleLiveEvent)

  async function handleResetKillSwitch() {
    setResetting(true)
    try {
      await api.resetKillSwitch()
      refreshAll()
    } catch (e) {
      setError(String(e))
    } finally {
      setResetting(false)
    }
  }

  async function handleEmergencyStop() {
    setStopping(true)
    try {
      await api.emergencyStop()
      refreshAll()
    } catch (e) {
      setError(String(e))
    } finally {
      setStopping(false)
    }
  }

  async function handlePromote(id: number) {
    setBusyModelId(id)
    try {
      await api.promoteModel(id)
      refreshAll()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusyModelId(null)
    }
  }

  async function handleReject(id: number) {
    setBusyModelId(id)
    try {
      await api.rejectModel(id)
      refreshAll()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusyModelId(null)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <span className="app-logo">◆</span>
          <h1>Trading Bot</h1>
        </div>
        <div className="app-header-right">
          <span className={`conn-indicator conn-${connectionState}`}>
            <span className="conn-dot" />
            {connectionState === 'connected' ? 'live' : connectionState === 'connecting' ? 'connecting…' : 'reconnecting…'}
          </span>
          <span className="app-clock num">{clock.toLocaleTimeString()}</span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <StatusBanner
        status={status}
        onResetKillSwitch={handleResetKillSwitch}
        resetting={resetting}
        onEmergencyStop={handleEmergencyStop}
        stopping={stopping}
      />

      <div className="grid">
        <EquityChart points={equity} onRangeChange={setEquityHours} />
        <PerformancePanel trades={trades} />
      </div>

      <TradesPanel openPositions={positions} recentTrades={trades} />

      <TrainingPanel
        models={models}
        pending={pending}
        onPromote={handlePromote}
        onReject={handleReject}
        busyModelId={busyModelId}
      />
    </div>
  )
}

export default App
