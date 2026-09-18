import type { Status } from '../api'

interface Props {
  status: Status | null
  onResetKillSwitch: () => void
  resetting: boolean
  onEmergencyStop: () => void
  stopping: boolean
}

export function StatusBanner({ status, onResetKillSwitch, resetting, onEmergencyStop, stopping }: Props) {
  if (!status) {
    return (
      <div className="hero hero-loading">
        <div className="skeleton skeleton-line" style={{ width: 220 }} />
      </div>
    )
  }

  const halted = status.trading_halted
  const nav = status.latest_nav

  return (
    <div className={`hero ${halted ? 'hero-halted' : 'hero-active'}`}>
      <div className="hero-top">
        <div className="hero-nav-block">
          <div className="hero-nav-label">
            <span className={`dot ${halted ? 'dot-red' : 'dot-green'}`} />
            Account NAV
          </div>
          <div className="hero-nav-value num">
            {nav !== null ? nav.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'}
          </div>
        </div>

        <div className="hero-meta">
          <MetaItem label="Mode">
            <span className={`badge ${status.environment === 'live' ? 'badge-live-mode' : 'badge-practice'}`}>
              {status.environment.toUpperCase()}
            </span>
          </MetaItem>
          <MetaItem label="Engine">
            <span className={status.engine_running ? 'text-green' : 'text-dim'}>
              {status.engine_running ? 'running' : 'stopped'}
            </span>
          </MetaItem>
          <MetaItem label="Trading">
            <span className={halted ? 'text-red' : 'text-green'}>{halted ? 'HALTED' : 'active'}</span>
          </MetaItem>
          {status.pending_promotions > 0 && (
            <MetaItem label="Pending">
              <span className="text-yellow">{status.pending_promotions} awaiting review</span>
            </MetaItem>
          )}
        </div>

        {halted ? (
          <button className="btn btn-resume" onClick={onResetKillSwitch} disabled={resetting}>
            {resetting ? 'Resuming…' : 'Resume trading'}
          </button>
        ) : (
          <button
            className="btn btn-emergency-stop"
            disabled={stopping}
            onClick={() => {
              if (window.confirm('Emergency stop: this halts trading and closes all open positions immediately. Continue?')) {
                onEmergencyStop()
              }
            }}
          >
            {stopping ? 'Stopping…' : 'Emergency Stop'}
          </button>
        )}
      </div>

      {status.instruments.length > 0 && (
        <div className="instrument-row">
          {status.instruments.map((inst) => (
            <div key={inst.instrument} className="instrument-chip">
              <span className="instrument-chip-name num">{inst.instrument}</span>
              {inst.live_model ? (
                <span className="text-accent instrument-chip-strategy">{inst.live_model.name}</span>
              ) : (
                <span className="text-dim instrument-chip-strategy">baseline</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MetaItem({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="hero-meta-item">
      <div className="hero-meta-label">{label}</div>
      <div className="hero-meta-value">{children}</div>
    </div>
  )
}
