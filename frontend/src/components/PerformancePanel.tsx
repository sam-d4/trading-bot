import type { Trade } from '../api'

interface Props {
  trades: Trade[]
}

function computeStats(closed: Trade[]) {
  const pnls = closed.map((t) => t.realized_pnl ?? 0)
  const wins = pnls.filter((p) => p > 0)
  const losses = pnls.filter((p) => p < 0)
  const winRate = pnls.length ? wins.length / pnls.length : 0
  const avgWin = wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : 0
  const avgLoss = losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0
  const grossWin = wins.reduce((a, b) => a + b, 0)
  const grossLoss = -losses.reduce((a, b) => a + b, 0)
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0
  const totalPnl = pnls.reduce((a, b) => a + b, 0)

  let currentStreak = 0
  let currentStreakType: 'win' | 'loss' | null = null
  let bestWinStreak = 0
  let worstLossStreak = 0
  let runningWin = 0
  let runningLoss = 0
  for (const p of pnls) {
    if (p > 0) {
      runningWin += 1
      runningLoss = 0
      bestWinStreak = Math.max(bestWinStreak, runningWin)
    } else if (p < 0) {
      runningLoss += 1
      runningWin = 0
      worstLossStreak = Math.max(worstLossStreak, runningLoss)
    } else {
      runningWin = 0
      runningLoss = 0
    }
  }
  if (pnls.length > 0) {
    const last = pnls[pnls.length - 1]
    currentStreakType = last > 0 ? 'win' : last < 0 ? 'loss' : null
    currentStreak = last > 0 ? runningWin : last < 0 ? runningLoss : 0
  }

  return { winRate, avgWin, avgLoss, profitFactor, totalPnl, tradeCount: pnls.length, bestWinStreak, worstLossStreak, currentStreak, currentStreakType }
}

export function PerformancePanel({ trades }: Props) {
  const closed = trades.filter((t) => t.status === 'closed')
  const stats = computeStats(closed)

  return (
    <div className="panel">
      <h2>Performance</h2>
      {stats.tradeCount === 0 ? (
        <div className="empty-state">No closed trades yet.</div>
      ) : (
        <>
          <div className="pnl-hero">
            <div className="pnl-hero-label">Total P&amp;L</div>
            <div className={`pnl-hero-value num ${stats.totalPnl >= 0 ? 'text-green' : 'text-red'}`}>
              {stats.totalPnl >= 0 ? '+' : ''}
              {stats.totalPnl.toFixed(2)}
            </div>
          </div>
          <div className="stats-grid">
            <Stat label="Closed trades" value={stats.tradeCount.toString()} />
            <Stat label="Win rate" value={`${(stats.winRate * 100).toFixed(1)}%`} />
            <Stat label="Profit factor" value={Number.isFinite(stats.profitFactor) ? stats.profitFactor.toFixed(2) : '∞'} />
            <Stat label="Avg win" value={`+${stats.avgWin.toFixed(2)}`} positive />
            <Stat label="Avg loss" value={stats.avgLoss.toFixed(2)} positive={false} />
            <Stat
              label="Current streak"
              value={stats.currentStreakType ? `${stats.currentStreak} ${stats.currentStreakType}` : '—'}
              positive={stats.currentStreakType === 'win' ? true : stats.currentStreakType === 'loss' ? false : undefined}
            />
            <Stat label="Best win streak" value={stats.bestWinStreak.toString()} />
            <Stat label="Worst loss streak" value={stats.worstLossStreak.toString()} />
          </div>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  const cls = positive === undefined ? '' : positive ? 'text-green' : 'text-red'
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value num ${cls}`}>{value}</div>
    </div>
  )
}
