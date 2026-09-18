# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal, multi-instrument FOREX trading bot against OANDA's v20 API (practice account by
default — `OANDA_ENVIRONMENT` in `.env` is the only switch to real money, and it's a manual edit
+ restart, never an in-app toggle). Python/FastAPI backend, React/TS dashboard, SQLite,
`stable-baselines3`/`gymnasium` for an RL strategy layer that self-improves on a schedule but is
gated behind a validation step and (by default) manual confirmation before anything trades real
money. There is no README — this file plus the module docstrings (unusually thorough in this
repo; read them before assuming behavior) are the documentation.

Pushed to a private GitHub repo (`sam-d4/trading-bot`, remote `origin`/`main`). The user has
asked that this repo stay updated as work progresses — commit and push meaningful chunks of
completed work (a feature, a fix, a finished phase) without needing to ask each time, rather
than batching everything into rare, large commits or leaving work uncommitted between sessions.
Still exercise normal judgment on commit granularity and message quality; this authorizes the
routine of committing/pushing itself, not skipping review of what's being staged.

## Commands

Python env uses `uv` (not plain pip) and a local `.venv`:

```bash
uv pip install --python .venv -e ".[dev]"       # base + dev deps (fast)
uv pip install --python .venv -e ".[dev,rl]"    # add torch/stable-baselines3 - needed for anything touching app/rl or the live engine's model hot-swap
```

Tests (pytest marker `slow` = actual PyTorch training, excluded by default via `addopts` in `pyproject.toml`):

```bash
.venv/bin/python -m pytest tests/ -q                       # fast suite only (~5-10s)
.venv/bin/python -m pytest tests/ -q -m slow                # slow RL pipeline/tournament tests (~1-2min)
.venv/bin/python -m pytest tests/test_position_sizing.py -q # single file
.venv/bin/python -m pytest tests/test_risk_manager.py::test_manual_trip_halts_regardless_of_drawdown -q  # single test
```

Run the app locally:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Boots with the baseline rule-based strategy on every pair in `TRADED_INSTRUMENTS_CSV` if
`OANDA_API_TOKEN`/`OANDA_ACCOUNT_ID` are unset in `.env` — that's a deliberate no-crash fallback,
not a bug, since the [rl] extra and OANDA creds are both optional at boot. Check `/healthz` and
`GET /api/status`. **Before restarting the live process, check `GET /api/positions` is empty**
(or that any open positions are accounted for) — the FX market doesn't pause for a redeploy.

DB migrations (Alembic; a `Base.metadata.create_all()` in `init_db()` is a dev-only fallback and
does NOT pick up column changes on existing tables — a schema change to `app/persistence/models.py`
needs a real migration):

```bash
.venv/bin/alembic revision --autogenerate -m "..."
.venv/bin/alembic upgrade head
```

Frontend (`frontend/`, separate `npm` project — `.venv` doesn't cover it):

```bash
npm run build   # tsc -b && vite build -> frontend/dist, which app/main.py mounts at "/" if it exists
npm run dev     # vite dev server; proxies /api and /ws to :8000 (see vite.config.ts) - point OANDA creds' backend at 8000 separately
```

Data pipeline / RL scripts (all read `.env` via `app/config/settings.py`; all need
`OANDA_API_TOKEN`/`OANDA_ACCOUNT_ID` set, most also want cached historical candles first):

```bash
.venv/bin/python scripts/check_oanda_connection.py                    # account summary + a live test order round-trip
.venv/bin/python scripts/fetch_historical_data.py --instruments EUR_USD GBP_USD USD_JPY AUD_USD --years 5
.venv/bin/python scripts/run_backtest.py --instrument EUR_USD
.venv/bin/python scripts/run_training.py --instrument EUR_USD --timesteps 100000 --contestants 4
```
`run_training.py` is what the background scheduler (`app/rl/scheduler.py`) runs automatically per
instrument on a staggered rotation — same tournament pipeline, run by hand.

### Environment gotchas specific to this machine/repo

- Only Python 3.7 is preinstalled here (too old — needs 3.11+). `uv python install 3.12` is
  dramatically faster than `brew install python@3.12`, which was seen compiling from source
  (OpenSSL et al.) for 20+ minutes on this box. Prefer `uv`.
- No Node preinstalled either; a prebuilt tarball from nodejs.org unpacked into
  `~/.local/opt/node-*/bin` (added to `PATH` for that shell) was faster than waiting on Homebrew
  here too.
- `numpy` is pinned `<2` in `pyproject.toml` on purpose: torch's last x86_64-macOS wheel (2.2.2)
  was built against NumPy 1.x's ABI and breaks silently under NumPy 2.x. Don't "fix" this pin
  without checking the target platform's torch build.
- `stable_baselines3` imports `matplotlib` (for plot logging this project never uses) at import
  time, which can crash or hang scanning system fonts on some macOS setups. `app/rl/__init__.py`
  and `app/strategy/rl_policy.py` both set `MPL_IGNORE_SYSTEM_FONTS=1` before that import as a
  fix — keep that if you touch either file's imports.
- `pandas-ta` was tried and dropped (stale release, hung under this project's pandas version).
  `app/data/features.py`'s indicators (EMA/RSI/ATR/z-score) are hand-rolled on purpose — don't
  reach for `pandas-ta` to "simplify" them.

## Architecture

### The one thing everything else depends on: one feature pipeline, live and training

`app/data/features.py`'s `compute_features()` is called identically by the live engine, the
backtest engine, and every RL training/validation path — this is what the module docstring means
by "no train/serve skew." Its output columns are NOT static: `feature_columns_for(instrument)` /
`infer_feature_columns(df)` derive them per-instrument, because each pair's cross-pair columns
name the *other* pairs (`xpair_gbp_usd_momentum` for EUR_USD, `xpair_eur_usd_momentum` for
GBP_USD, etc. — see `related_instruments()`). `TradingEnv` and `RLPolicyStrategy` both take/infer
`feature_columns` explicitly rather than importing a fixed list, and
`persistence.models.ModelVersion.instrument`/`.lookback` record what a saved model actually
expects, so loading one later never has to guess its input shape. If you add a feature column,
everything downstream picks it up automatically **as long as every caller still computes
`related_candles`/`carry_differential` consistently** — a live/training mismatch here is a silent
shape bug, not a crash, so don't skip that when adding a new call site.

### Trading engine (`app/core/engine.py`)

One `TradingEngine` trades every instrument in `settings.traded_instruments` concurrently, each
with its own independently-promotable strategy (`dict[instrument, Strategy]`), sharing:
- one OANDA price stream connection (comma-joined instruments — OANDA supports this natively),
  covering both traded pairs and whatever's only needed as cross-pair context for them
- one account-wide `RiskManager` kill-switch (a bad day on one pair can halt all of them — that's
  intentional, it protects the account, not a position)
- independent per-instrument bar aggregation, feature computation, and strategy `decide()` calls
- a per-instrument hot-swap poll (`_poll_live_model_swap`) that loads a newly-promoted model for
  *that* instrument without restarting the process or touching the others

`app/execution/order_manager.py` refuses orders while halted (duplicating the risk manager's own
check as defense in depth) and enforces "no pyramiding" (a same-direction repeat signal is a
no-op; a reversal closes then reopens). `app/execution/reconciliation.py` periodically re-syncs
against OANDA's own account state, which is always the source of truth — the local DB mirrors it.

### Risk & sizing (`app/risk/`)

- `RiskManager`: two independent drawdown limits (daily/overall) off account NAV (balance +
  unrealized P&L), checked both pre-trade and on an independent poll. Never auto-resets — only an
  explicit dashboard action clears a halt, including the manual "Emergency Stop" button
  (`POST /api/kill-switch/trip`), which reuses the exact same halt mechanism, not a second one.
- `position_sizing.py`: fixed-fractional risk off ATR-based stop distance, capped by
  `max_leverage`. Has an `account_to_quote_rate` parameter because the account's currency and an
  instrument's quote currency aren't always the same (this was a real, previously-shipped bug on
  a GBP account trading EUR_USD/USD-quoted pairs) — `resolve_account_to_quote_rate()` resolves it
  from whatever cross-rate is currently tracked in the engine's own bars, falling back to `1.0`
  with a loud warning rather than crashing the live loop.
- `RiskLimits.from_settings()` divides `risk_per_trade_pct` by the number of traded instruments,
  so adding pairs spreads the same aggregate risk budget rather than multiplying it.

### RL self-improvement (`app/rl/`)

Cold-start is never a blank policy: `pretrain.py` behavior-clones a fresh PPO policy against
`app/strategy/baselines.py`'s rule-based strategies before any RL training happens
("cold-start knowledge"). `train.py` then fine-tunes with PPO. `env.py`'s `TradingEnv` supports
two reward shapes — `reward_mode="pnl"` (raw net return) vs the default `"differential_sharpe"`
(Moody & Saffell's per-step Sharpe-tracking reward, which aligns training with the Sharpe-based
gate below and, unlike raw P&L, penalizes volatility). Both modes subtract a small standing
flat-position penalty (`DEFAULT_FLAT_PENALTY_BPS`) — without it, PPO reliably converges to "never
trade" once it discovers most trades lose to costs, since staying flat forever scores exactly 0
and nothing pushes it to reconsider. None of this reward shaping leaks into
`app/backtest/engine.py`'s `run_backtest`, which is what the validation gate and every downstream
comparison actually scores — real P&L net of real costs only.

`tournament.py` trains N independently-seeded contestants (different seed + entropy coefficient
each) and ranks them purely by ending virtual balance on a held-out window — RL training is
seed-sensitive enough that one run's outcome isn't representative. Only the winner proceeds to
`evaluate.py`'s validation gate (must beat buy-and-hold, the baseline library, AND the current
live model on Sharpe); every eliminated contestant is still registered in
`persistence.models.ModelVersion` with `status=rejected` and its tournament rank, for audit
visibility in the dashboard's model history — the tournament is a selection step, not the safety
gate. A passing candidate becomes `status=validated` and sits on the dashboard's "pending
promotion" card; only an explicit `POST /api/models/{id}/promote` (or
`require_manual_confirmation_to_promote=false`, which is not the default) flips it to `live`.

`scheduler.py` runs this per-instrument on a staggered rotation (one job per traded instrument,
each firing every `interval_hours * len(traded_instruments)`, offset so only one tournament's
PPO training runs at a time) rather than retraining every pair on every tick.

### Dashboard (`app/api/`, `frontend/src/`)

REST (`routes_dashboard.py`, `routes_control.py`) for initial load and mutations; a WebSocket
relay (`ws.py` + `app/core/events.py`'s in-process pub/sub — no external broker, this is single-
instance) pushes live trade/equity/kill-switch/model events. `GET /api/status` returns a
per-instrument breakdown (`instruments: [...]`), not a single instrument/model — the frontend's
`StatusBanner` renders one chip per traded pair. Basic auth (`api/auth.py`) is a no-op until
`DASHBOARD_BASIC_AUTH_USER` is set — fine for local dev, required before exposing this beyond
localhost. Validation-gate metrics are stored as JSON in
`ModelVersion.validation_metrics_json`; `persistence/repo.py`'s `_json_safe()` scrubs
`inf`/`-inf`/`nan` before serializing — Python's `json.dumps` emits the bare tokens
Infinity/NaN by default, which is invalid per the JSON spec and made the dashboard silently fail
to render (caught in this repo's own history) whenever a strategy had zero losing trades
(`profit_factor = inf`, which `buy_and_hold` hits constantly). Don't remove that scrub.
