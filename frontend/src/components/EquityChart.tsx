import { useMemo, useState } from 'react'
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EquityPoint } from '../api'

interface Props {
  points: EquityPoint[]
  onRangeChange: (hours: number) => void
}

const RANGES: { label: string; hours: number }[] = [
  { label: '1D', hours: 24 },
  { label: '1W', hours: 24 * 7 },
  { label: '1M', hours: 24 * 30 },
  { label: 'ALL', hours: 24 * 365 * 5 },
]

export function EquityChart({ points, onRangeChange }: Props) {
  const [activeRange, setActiveRange] = useState(1) // default 1W, matches api.ts default

  const data = useMemo(
    () =>
      points.map((p) => ({
        time: p.taken_at,
        label: new Date(p.taken_at).toLocaleString(),
        nav: p.nav,
      })),
    [points],
  )

  const trendUp = data.length > 1 && data[data.length - 1].nav >= data[0].nav
  const lineColor = trendUp ? 'var(--green)' : 'var(--red)'

  return (
    <div className="panel panel-chart">
      <div className="panel-header">
        <h2>Equity Curve</h2>
        <div className="range-tabs">
          {RANGES.map((r, i) => (
            <button
              key={r.label}
              className={`range-tab ${i === activeRange ? 'range-tab-active' : ''}`}
              onClick={() => {
                setActiveRange(i)
                onRangeChange(r.hours)
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
      {data.length === 0 ? (
        <div className="empty-state">No equity snapshots yet — waiting on the first NAV poll.</div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="navFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity={0.28} />
                <stop offset="100%" stopColor={lineColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tick={false} axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
            <YAxis
              domain={['auto', 'auto']}
              width={72}
              tick={{ fontSize: 11, fill: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<ChartTooltip />} />
            <Area type="monotone" dataKey="nav" stroke={lineColor} strokeWidth={1.75} fill="url(#navFill)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

interface TooltipProps {
  active?: boolean
  payload?: { payload: { nav: number; label: string } }[]
}

function ChartTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-value num">{point.nav.toFixed(2)}</div>
      <div className="chart-tooltip-time">{point.label}</div>
    </div>
  )
}
