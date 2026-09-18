"""Live trading loop orchestrator - trades several instruments concurrently (one strategy/model
per instrument, independently promotable/hot-swappable), sharing one account-level risk manager.

Five concurrent responsibilities, matching the plan's data-flow design:
  1. consume one shared price stream covering every traded instrument (see
     app/broker/oanda_stream.py - OANDA natively supports multiple instruments per connection),
     aggregate each into its own bars, and on each completed bar for a TRADED instrument run
     that instrument's own strategy and (if it signals a trade) the order manager. Every traded
     instrument also serves as cross-pair context for the others' features - no separate
     "context-only" instrument set needed once all of app/data/features.py's INSTRUMENT_UNIVERSE
     is being traded.
  2. an independent periodic NAV poll that feeds the risk manager's kill-switch (one, account-
     wide, shared across every instrument - a bad day on one pair can halt trading on all of
     them, which is the point: the kill-switch protects the account, not a single position)
  3. a periodic reconciliation pass against OANDA's own account state (source of truth)
  4. a periodic per-instrument check for a newly-promoted live model (app/rl/registry.py flips a
     ModelVersion to LIVE only after the manual-confirmation step) - each instrument starts on
     its own rule-based baseline strategy and hot-swaps to its own RL policy independently, the
     moment one is promoted for THAT instrument, with no restart required.
  5. a periodic refresh of the real carry (financing) rate for every traded instrument at once
     (OANDA's financing-rate endpoint already accepts a list) - see
     app/broker/oanda_client.py's get_financing_rates.

Feeds app/data/features.py's compute_features with the exact same related_candles/
carry_differential shape that scripts/run_backtest.py and scripts/run_training.py use per
instrument - that consistency (not just "the live engine also calls compute_features") is what
keeps a trained model's expected observation shape matching what it's actually fed live. See
features.py's feature_columns_for.

Position sizing risk (equity * risk_per_trade_pct) is deliberately scaled down per-instrument by
RiskLimits.from_settings based on how many instruments are traded concurrently, so trading N
pairs doesn't multiply worst-case same-day loss by N - see that function's docstring.

NOT YET VERIFIED end-to-end against a live OANDA stream with multiple TRADED instruments (this
started as a single-instrument design, generalized once cross-pair features made multi-
instrument streaming necessary anyway, then again to trade every streamed instrument rather
than just the primary one) - re-verify live once run.
"""

import asyncio
import collections
import datetime as dt

import pandas as pd
import structlog

from app.broker.models import Tick
from app.broker.oanda_client import OandaClient, OandaClientError
from app.broker.oanda_stream import OandaPriceStream
from app.config.settings import Settings
from app.core.events import Event, EventType, event_bus
from app.data.features import compute_features, related_instruments
from app.execution.order_manager import OrderManager
from app.execution.reconciliation import reconcile
from app.notifications.alerts import AlertSender
from app.persistence import repo
from app.persistence.db import SessionLocal
from app.risk.manager import RiskManager
from app.risk.position_sizing import resolve_account_to_quote_rate
from app.strategy.base import Signal, Strategy

log = structlog.get_logger(__name__)

# Must comfortably exceed compute_features' internal indicator warmup so the live feature
# window is never truncated.
BAR_HISTORY_LEN = 400
CARRY_REFRESH_SECONDS = 6 * 3600  # financing rates change occasionally, not every bar


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        client: OandaClient,
        strategies: dict[str, Strategy],
        risk_manager: RiskManager,
        order_manager: OrderManager,
        alert_sender: AlertSender,
        model_version_ids: dict[str, int | None] | None = None,
    ):
        self._settings = settings
        self._client = client
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._alerts = alert_sender
        self._strategies = dict(strategies)
        self._traded_instruments = list(strategies.keys())
        self._model_version_ids: dict[str, int | None] = model_version_ids or dict.fromkeys(self._traded_instruments)

        # Every traded instrument also doubles as cross-pair context for the others, so the set
        # of instruments to actually stream/track bars for is the union of what's traded and
        # what each traded instrument needs as context - in the default all-4-traded config
        # these are the same set, but this stays correct if fewer than the full universe trade.
        context_needed = {inst for t in self._traded_instruments for inst in related_instruments(t)}
        self._all_instruments = list({*self._traded_instruments, *context_needed})

        self._bars: dict[str, collections.deque] = {
            inst: collections.deque(maxlen=BAR_HISTORY_LEN) for inst in self._all_instruments
        }
        self._current_bar: dict[str, dict | None] = dict.fromkeys(self._all_instruments)
        self._bucket_start: dict[str, dt.datetime | None] = dict.fromkeys(self._all_instruments)
        self._stream_last_tick_at: dt.datetime | None = None
        self._carry_differential: dict[str, float] = dict.fromkeys(self._traded_instruments, 0.0)

    async def run(self) -> None:
        stream = OandaPriceStream(self._settings, self._all_instruments)
        tick_queue = stream.ticks()

        log.info(
            "engine_started",
            traded_instruments=self._traded_instruments,
            strategies={inst: s.name for inst, s in self._strategies.items()},
        )
        try:
            await asyncio.gather(
                self._consume_ticks(tick_queue),
                self._poll_kill_switch(),
                self._poll_reconciliation(),
                self._poll_live_model_swap(),
                self._poll_carry_rate(),
            )
        finally:
            stream.stop()

    # --- bar aggregation + strategy decisions -----------------------------------------------

    async def _consume_ticks(self, tick_queue: asyncio.Queue[Tick]) -> None:
        disconnect_alerted = False
        while True:
            try:
                tick = await asyncio.wait_for(tick_queue.get(), timeout=30.0)
            except TimeoutError:
                if self._stream_last_tick_at is not None and not disconnect_alerted:
                    gap = dt.datetime.now(dt.timezone.utc) - self._stream_last_tick_at
                    if gap > dt.timedelta(minutes=2):
                        self._alerts.send(
                            "Price stream disconnected",
                            f"No ticks for {', '.join(self._all_instruments)} in over 2 minutes.",
                        )
                        disconnect_alerted = True
                continue

            disconnect_alerted = False
            self._stream_last_tick_at = dt.datetime.now(dt.timezone.utc)
            self._on_tick(tick)

    def _on_tick(self, tick: Tick) -> None:
        if tick.instrument not in self._bars:
            return  # a stray instrument we didn't ask to stream - ignore defensively

        # OANDA ticks carry nanosecond precision; datetime only supports microseconds, so floor
        # before converting rather than letting pandas warn on every single tick.
        tick_time = pd.Timestamp(tick.time).floor("us").to_pydatetime()
        bucket = tick_time.replace(
            second=(tick_time.second // self._settings.live_bar_seconds) * self._settings.live_bar_seconds,
            microsecond=0,
        )

        inst = tick.instrument
        if self._bucket_start[inst] is None:
            self._bucket_start[inst] = bucket
            self._current_bar[inst] = self._new_bar(tick.mid)
            return

        if bucket != self._bucket_start[inst]:
            self._finalize_bar(inst)
            self._bucket_start[inst] = bucket
            self._current_bar[inst] = self._new_bar(tick.mid)
            return

        bar = self._current_bar[inst]
        bar["high"] = max(bar["high"], tick.mid)
        bar["low"] = min(bar["low"], tick.mid)
        bar["close"] = tick.mid
        bar["volume"] += 1

    def _new_bar(self, price: float) -> dict:
        return {"open": price, "high": price, "low": price, "close": price, "volume": 1}

    def _finalize_bar(self, instrument: str) -> None:
        current = self._current_bar[instrument]
        bucket_start = self._bucket_start[instrument]
        if current is None or bucket_start is None:
            return
        self._bars[instrument].append({"time": bucket_start, **current})

        if instrument not in self._traded_instruments:
            return  # tracked only as cross-pair context for a traded instrument, never itself traded

        if len(self._bars[instrument]) < BAR_HISTORY_LEN // 2:
            return  # not enough history yet for reliable indicators

        related_dfs = {
            inst: pd.DataFrame(self._bars[inst])
            for inst in related_instruments(instrument)
            if inst in self._bars and len(self._bars[inst]) >= 5  # a handful of bars is enough for merge_asof context
        }

        df = pd.DataFrame(self._bars[instrument])
        carry = self._carry_differential.get(instrument, 0.0)
        features = compute_features(df, related_candles=related_dfs, carry_differential=carry)
        if features.empty:
            return

        latest = features.iloc[-1]
        signal = self._strategies[instrument].decide(latest)
        self._act_on_signal(instrument, signal, latest)

    def _act_on_signal(self, instrument: str, signal: Signal, features_row: pd.Series) -> None:
        try:
            summary = self._client.get_account_summary()
        except OandaClientError as exc:
            log.error("account_summary_fetch_failed_skipping_decision", instrument=instrument, error=str(exc))
            return

        direction = {Signal.LONG: 1, Signal.SHORT: -1, Signal.FLAT: 0}[signal]
        atr = features_row["atr_pct"] * features_row["close"]
        account_to_quote_rate = self._resolve_account_to_quote_rate(instrument, summary.currency)

        with SessionLocal() as session:
            self._order_manager.execute_signal(
                session,
                instrument=instrument,
                direction=direction,
                price=features_row["close"],
                atr=atr,
                equity=summary.nav,
                strategy_name=self._strategies[instrument].name,
                model_version_id=self._model_version_ids.get(instrument),
                account_to_quote_rate=account_to_quote_rate,
            )

    def _resolve_account_to_quote_rate(self, instrument: str, account_currency: str) -> float:
        """See app/risk/position_sizing.py's module docstring for the currency-mismatch bug this
        fixes. Falls back to 1.0 (no conversion) with a loud warning if the needed cross-rate
        isn't available from currently-tracked bars, rather than crashing the live loop over it."""
        quote_currency = instrument.split("_")[1]
        rate_lookup = {inst: bars[-1]["close"] for inst, bars in self._bars.items() if bars}
        try:
            return resolve_account_to_quote_rate(account_currency, quote_currency, rate_lookup)
        except ValueError as exc:
            log.warning("account_to_quote_rate_unavailable_falling_back_to_1", instrument=instrument, error=str(exc))
            return 1.0

    # --- kill-switch polling (one, account-wide, shared across every traded instrument) --------

    async def _poll_kill_switch(self) -> None:
        while True:
            await asyncio.sleep(self._settings.kill_switch_poll_seconds)
            try:
                summary = self._client.get_account_summary()
            except OandaClientError as exc:
                log.error("kill_switch_poll_fetch_failed", error=str(exc))
                continue

            now = dt.datetime.now(dt.timezone.utc)
            with SessionLocal() as session:
                repo.record_equity_snapshot(
                    session, nav=summary.nav, balance=summary.balance, unrealized_pnl=summary.unrealized_pl
                )
                event_bus.publish(
                    Event(EventType.EQUITY_UPDATE, {"nav": summary.nav, "unrealized_pnl": summary.unrealized_pl})
                )

                check = self._risk_manager.check(summary.nav, now)
                if check.newly_tripped:
                    repo.record_kill_switch_event(
                        session, event="tripped", reason=check.reason or "unknown", nav_at_event=summary.nav
                    )
                    if self._settings.flatten_on_kill_switch:
                        self._order_manager.flatten_all(session)
                    event_bus.publish(Event(EventType.KILL_SWITCH_TRIPPED, {"reason": check.reason}))
                    self._alerts.send("Kill-switch tripped", check.reason or "unknown reason")

    # --- reconciliation ------------------------------------------------------------------------

    async def _poll_reconciliation(self) -> None:
        while True:
            await asyncio.sleep(self._settings.reconciliation_interval_seconds)
            with SessionLocal() as session:
                reconcile(session, self._client)

    # --- carry (financing) rate refresh ---------------------------------------------------------

    async def _poll_carry_rate(self) -> None:
        while True:
            try:
                rates = self._client.get_financing_rates(self._traded_instruments)
                self._carry_differential.update(rates)
                log.info("carry_rates_refreshed", rates={k: round(v, 4) for k, v in rates.items()})
            except OandaClientError as exc:
                log.warning("carry_rate_refresh_failed", error=str(exc))
            await asyncio.sleep(CARRY_REFRESH_SECONDS)

    # --- live model hot-swap (per instrument, independently) -----------------------------------

    async def _poll_live_model_swap(self) -> None:
        while True:
            await asyncio.sleep(self._settings.live_model_poll_seconds)
            with SessionLocal() as session:
                live_by_instrument = repo.current_live_models_by_instrument(session, self._traded_instruments)

            for instrument, live_mv in live_by_instrument.items():
                if live_mv.id == self._model_version_ids.get(instrument):
                    continue  # nothing newly promoted for this instrument, or already running it

                if not live_mv.artifact_path:
                    log.error("live_model_missing_artifact_path", instrument=instrument, model_id=live_mv.id)
                    continue

                try:
                    # imported here (not at module top) to avoid importing torch/stable-baselines3
                    # for the common baseline-only path when no RL model has ever been promoted
                    from app.data.features import feature_columns_for
                    from app.strategy.rl_policy import RLPolicyStrategy

                    new_strategy = RLPolicyStrategy.load(
                        live_mv.artifact_path,
                        feature_columns_for(instrument),
                        lookback=live_mv.lookback or self._settings.rl_lookback,
                        model_version_name=live_mv.name,
                    )
                except Exception as exc:  # noqa: BLE001 - a bad artifact must not crash the live loop
                    log.error("live_model_load_failed", instrument=instrument, model_id=live_mv.id, error=str(exc))
                    continue

                log.info(
                    "live_model_swapped",
                    instrument=instrument,
                    model_id=live_mv.id,
                    name=live_mv.name,
                    previous=self._strategies[instrument].name,
                )
                self._strategies[instrument] = new_strategy
                self._model_version_ids[instrument] = live_mv.id
                event_bus.publish(
                    Event(EventType.MODEL_PROMOTED, {"instrument": instrument, "model_id": live_mv.id, "name": live_mv.name})
                )
