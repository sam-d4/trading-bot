import { useMemo, useState } from 'react'
import type { ModelVersionSummary, PendingPromotion } from '../api'

interface Props {
  models: ModelVersionSummary[]
  pending: PendingPromotion[]
  onPromote: (id: number) => void
  onReject: (id: number) => void
  busyModelId: number | null
}

interface StrategyMetrics {
  sharpe: number
  total_return_pct: number
  max_drawdown_pct: number
  trade_count: number
}

interface PassedMetrics {
  candidate: StrategyMetrics
  comparison: {
    buy_and_hold: StrategyMetrics
    best_baseline: StrategyMetrics & { name: string }
    live_model: StrategyMetrics | null
    library_strategies?: Record<string, StrategyMetrics>
  }
}

interface RejectedMetrics {
  candidate: StrategyMetrics
  reasons: string[]
}

function parseMetrics(json: string | null): PassedMetrics | RejectedMetrics | null {
  if (!json) return null
  try {
    return JSON.parse(json)
  } catch {
    return null
  }
}

function fmtNum(n: number | undefined): string {
  if (n === undefined || n === null) return '—'
  return n.toFixed(3)
}

function sharpeClass(n: number | undefined): string {
  if (n === undefined) return ''
  return n > 0 ? 'text-green' : n < 0 ? 'text-red' : ''
}

export function TrainingPanel({ models, pending, onPromote, onReject, busyModelId }: Props) {
  const [instrumentFilter, setInstrumentFilter] = useState<string>('all')

  const instruments = useMemo(() => {
    const set = new Set(models.map((m) => m.instrument).filter((i): i is string => !!i))
    return ['all', ...Array.from(set).sort()]
  }, [models])

  const filteredModels = instrumentFilter === 'all' ? models : models.filter((m) => m.instrument === instrumentFilter)

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Self-Improvement</h2>
      </div>

      {pending.length > 0 && (
        <div className="promotion-queue">
          <h3>
            Pending promotion <span className="count-badge count-badge-yellow">{pending.length}</span>
          </h3>
          {pending.map((p) => {
            const metrics = parseMetrics(p.validation_metrics)
            const passed = metrics && 'comparison' in metrics ? (metrics as PassedMetrics) : null
            return (
              <div key={p.id} className="promotion-card">
                <div className="promotion-card-header">
                  {p.instrument && <span className="badge badge-practice">{p.instrument}</span>}
                  <strong>{p.name}</strong>
                  <span className="badge badge-validated">validated</span>
                </div>
                {passed && (
                  <div className="table-scroll">
                    <table className="table table-compact">
                      <thead>
                        <tr>
                          <th></th>
                          <th>Sharpe</th>
                          <th>Return %</th>
                          <th>Max DD %</th>
                          <th>Trades</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="row-highlight">
                          <td>Candidate</td>
                          <td className={`num ${sharpeClass(passed.candidate.sharpe)}`}>{fmtNum(passed.candidate.sharpe)}</td>
                          <td className="num">{fmtNum(passed.candidate.total_return_pct)}</td>
                          <td className="num">{fmtNum(passed.candidate.max_drawdown_pct)}</td>
                          <td className="num">{passed.candidate.trade_count}</td>
                        </tr>
                        <tr>
                          <td className="text-dim">Buy &amp; hold</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.buy_and_hold?.sharpe)}</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.buy_and_hold?.total_return_pct)}</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.buy_and_hold?.max_drawdown_pct)}</td>
                          <td className="num text-dim">{passed.comparison.buy_and_hold?.trade_count ?? '—'}</td>
                        </tr>
                        <tr>
                          <td className="text-dim">Best baseline ({passed.comparison.best_baseline?.name})</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.best_baseline?.sharpe)}</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.best_baseline?.total_return_pct)}</td>
                          <td className="num text-dim">{fmtNum(passed.comparison.best_baseline?.max_drawdown_pct)}</td>
                          <td className="num text-dim">{passed.comparison.best_baseline?.trade_count ?? '—'}</td>
                        </tr>
                        {passed.comparison.live_model && (
                          <tr>
                            <td className="text-dim">Current live model</td>
                            <td className="num text-dim">{fmtNum(passed.comparison.live_model.sharpe)}</td>
                            <td className="num text-dim">{fmtNum(passed.comparison.live_model.total_return_pct)}</td>
                            <td className="num text-dim">{fmtNum(passed.comparison.live_model.max_drawdown_pct)}</td>
                            <td className="num text-dim">{passed.comparison.live_model.trade_count}</td>
                          </tr>
                        )}
                        {passed.comparison.library_strategies &&
                          Object.entries(passed.comparison.library_strategies).map(([name, m]) => (
                            <tr key={name}>
                              <td className="text-dim">{name} (reference only)</td>
                              <td className="num text-dim">{fmtNum(m.sharpe)}</td>
                              <td className="num text-dim">{fmtNum(m.total_return_pct)}</td>
                              <td className="num text-dim">{fmtNum(m.max_drawdown_pct)}</td>
                              <td className="num text-dim">{m.trade_count}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <div className="promotion-actions">
                  <button className="btn btn-confirm" disabled={busyModelId === p.id} onClick={() => onPromote(p.id)}>
                    {busyModelId === p.id ? 'Working…' : 'Confirm & go live'}
                  </button>
                  <button className="btn btn-reject" disabled={busyModelId === p.id} onClick={() => onReject(p.id)}>
                    Reject
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div className="panel-header panel-header-tight">
        <h3 className="no-margin">Model history</h3>
        {instruments.length > 2 && (
          <div className="filter-tabs">
            {instruments.map((inst) => (
              <button
                key={inst}
                className={`filter-tab ${instrumentFilter === inst ? 'filter-tab-active' : ''}`}
                onClick={() => setInstrumentFilter(inst)}
              >
                {inst}
              </button>
            ))}
          </div>
        )}
      </div>
      {filteredModels.length === 0 ? (
        <div className="empty-state">No models trained yet.</div>
      ) : (
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Name</th>
                <th>Kind</th>
                <th>Status</th>
                <th>Sharpe</th>
                <th>Created</th>
                <th>Promoted</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m) => {
                const metrics = parseMetrics(m.validation_metrics)
                const reasons = metrics && 'reasons' in metrics ? (metrics as RejectedMetrics).reasons : null
                return (
                  <tr key={m.id}>
                    <td className="num">{m.instrument ?? '—'}</td>
                    <td>
                      {m.name}
                      {reasons && reasons.length > 0 && (
                        <div className="rejection-note" title={reasons.join('; ')}>
                          {reasons[0]}
                          {reasons.length > 1 ? ` (+${reasons.length - 1} more)` : ''}
                        </div>
                      )}
                    </td>
                    <td className="text-dim">{m.kind}</td>
                    <td>
                      <span className={`badge badge-${m.status}`}>{m.status}</span>
                    </td>
                    <td className={`num ${sharpeClass(metrics?.candidate.sharpe)}`}>{fmtNum(metrics?.candidate.sharpe)}</td>
                    <td>{new Date(m.created_at).toLocaleString()}</td>
                    <td>{m.promoted_at ? new Date(m.promoted_at).toLocaleString() : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
