import { useEffect, useMemo, useRef, useState } from 'react'
import type { Trade } from '../api'

interface Props {
  openPositions: Trade[]
  recentTrades: Trade[]
}

type Filter = 'all' | 'open' | 'closed'

function fmt(n: number | null): string {
  if (n === null) return '—'
  return n.toFixed(5)
}

function pnlClass(pnl: number | null): string {
  if (pnl === null) return ''
  return pnl >= 0 ? 'text-green' : 'text-red'
}

function DirectionTag({ direction }: { direction: 'long' | 'short' }) {
  const isLong = direction === 'long'
  return <span className={`direction-tag ${isLong ? 'text-green' : 'text-red'}`}>{isLong ? '▲ LONG' : '▼ SHORT'}</span>
}

export function TradesPanel({ openPositions, recentTrades }: Props) {
  const [filter, setFilter] = useState<Filter>('all')
  const [instrumentFilter, setInstrumentFilter] = useState<string>('all')
  const [flashId, setFlashId] = useState<number | null>(null)
  const lastTopId = useRef<number | null>(null)

  useEffect(() => {
    const topId = recentTrades[0]?.id ?? null
    if (topId !== null && topId !== lastTopId.current && lastTopId.current !== null) {
      setFlashId(topId)
      const timer = setTimeout(() => setFlashId(null), 2200)
      return () => clearTimeout(timer)
    }
    lastTopId.current = topId
  }, [recentTrades])

  const instruments = useMemo(
    () => ['all', ...Array.from(new Set(recentTrades.map((t) => t.instrument))).sort()],
    [recentTrades],
  )

  const filtered = recentTrades.filter(
    (t) => (filter === 'all' || t.status === filter) && (instrumentFilter === 'all' || t.instrument === instrumentFilter),
  )

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Live Trading</h2>
      </div>

      <h3>
        Open positions <span className="count-badge">{openPositions.length}</span>
      </h3>
      {openPositions.length === 0 ? (
        <div className="empty-state">No open positions.</div>
      ) : (
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Direction</th>
                <th>Units</th>
                <th>Entry</th>
                <th>Opened</th>
                <th>Strategy</th>
              </tr>
            </thead>
            <tbody>
              {openPositions.map((t) => (
                <tr key={t.id}>
                  <td className="num">{t.instrument}</td>
                  <td>
                    <DirectionTag direction={t.direction} />
                  </td>
                  <td className="num">{t.units.toFixed(0)}</td>
                  <td className="num">{fmt(t.entry_price)}</td>
                  <td>{new Date(t.opened_at).toLocaleString()}</td>
                  <td className="text-dim">{t.strategy_name}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel-header panel-header-tight">
        <h3 className="no-margin">Trade history</h3>
        <div className="trade-history-filters">
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
          <div className="filter-tabs">
            {(['all', 'open', 'closed'] as Filter[]).map((f) => (
              <button
                key={f}
                className={`filter-tab ${filter === f ? 'filter-tab-active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>
      {filtered.length === 0 ? (
        <div className="empty-state">{recentTrades.length === 0 ? 'No trades yet.' : 'No trades match this filter.'}</div>
      ) : (
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Direction</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&amp;L</th>
                <th>Status</th>
                <th>Strategy / model</th>
                <th>Opened</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => (
                <tr key={t.id} className={t.id === flashId ? 'row-flash' : ''}>
                  <td className="num">{t.instrument}</td>
                  <td>
                    <DirectionTag direction={t.direction} />
                  </td>
                  <td className="num">{fmt(t.entry_price)}</td>
                  <td className="num">{fmt(t.exit_price)}</td>
                  <td className={`num ${pnlClass(t.realized_pnl)}`}>
                    {t.realized_pnl !== null ? t.realized_pnl.toFixed(2) : '—'}
                  </td>
                  <td>
                    <span className={`status-pill status-pill-${t.status}`}>{t.status}</span>
                  </td>
                  <td className="text-dim">
                    {t.strategy_name}
                    {t.model_version_id !== null ? ` (model #${t.model_version_id})` : ''}
                  </td>
                  <td>{new Date(t.opened_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
