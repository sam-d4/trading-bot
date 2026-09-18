export interface Trade {
  id: number
  instrument: string
  direction: 'long' | 'short'
  units: number
  entry_price: number
  exit_price: number | null
  opened_at: string
  closed_at: string | null
  realized_pnl: number | null
  status: 'open' | 'closed'
  strategy_name: string
  model_version_id: number | null
}

export interface EquityPoint {
  taken_at: string
  nav: number
  balance: number
  unrealized_pnl: number
}

export interface ModelVersionSummary {
  id: number
  name: string
  kind: string
  instrument: string | null
  status: 'candidate' | 'validated' | 'live' | 'rejected' | 'retired'
  created_at: string
  promoted_at: string | null
  validation_metrics: string | null
}

export interface PendingPromotion {
  id: number
  name: string
  instrument: string | null
  validation_metrics: string | null
}

export interface InstrumentStatus {
  instrument: string
  strategy_name: string | null
  live_model: { id: number; name: string } | null
}

export interface Status {
  environment: 'practice' | 'live'
  engine_running: boolean
  trading_halted: boolean
  latest_nav: number | null
  pending_promotions: number
  instruments: InstrumentStatus[]
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path)
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`)
  return resp.json() as Promise<T>
}

async function postJson<T>(path: string): Promise<T> {
  const resp = await fetch(path, { method: 'POST' })
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`)
  return resp.json() as Promise<T>
}

export const api = {
  status: () => getJson<Status>('/api/status'),
  trades: (limit = 100) => getJson<Trade[]>(`/api/trades?limit=${limit}`),
  positions: () => getJson<Trade[]>('/api/positions'),
  equity: (hours = 24 * 7) => getJson<EquityPoint[]>(`/api/equity?hours=${hours}`),
  models: () => getJson<ModelVersionSummary[]>('/api/models'),
  pendingPromotions: () => getJson<PendingPromotion[]>('/api/models/pending'),
  resetKillSwitch: () => postJson<{ status: string; nav: number }>('/api/kill-switch/reset'),
  emergencyStop: () => postJson<{ status: string; nav: number }>('/api/kill-switch/trip'),
  promoteModel: (id: number) => postJson<{ status: string; model_id: number }>(`/api/models/${id}/promote`),
  rejectModel: (id: number) => postJson<{ status: string; model_id: number }>(`/api/models/${id}/reject`),
}
