"""Drawdown kill-switch. Two independent limits (daily, overall), both computed off account NAV
(balance + unrealized P&L) since an open position moving against you is real risk even before
it's realized - checked both pre-trade and on a periodic monitor tick independent of new orders.

On trip: halts all new orders (enforced again, independently, in execution/order_manager.py -
defense in depth) and, if configured, flattens open positions. Never auto-resets - resuming
requires an explicit call, which in the running system only happens via a manual dashboard
action, so a human consciously reviews what happened before trading resumes.
"""

import datetime as dt
from dataclasses import dataclass, field

from app.risk.limits import RiskLimits

DAILY_ROLLOVER_HOUR_UTC = 22  # matches OANDA's own convention for what counts as a "trading day"


def trading_day_key(now: dt.datetime) -> dt.date:
    shifted = now.astimezone(dt.timezone.utc) - dt.timedelta(hours=DAILY_ROLLOVER_HOUR_UTC)
    return shifted.date()


@dataclass
class KillSwitchState:
    high_water_mark: float
    daily_baseline_nav: float
    daily_key: dt.date
    halted: bool = False
    halt_reason: str | None = None


@dataclass
class KillSwitchCheck:
    halted: bool
    newly_tripped: bool
    reason: str | None = None


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self._limits = limits
        self._state: KillSwitchState | None = None

    def initialize(self, nav: float, now: dt.datetime) -> None:
        self._state = KillSwitchState(
            high_water_mark=nav, daily_baseline_nav=nav, daily_key=trading_day_key(now)
        )

    @property
    def halted(self) -> bool:
        return self._state.halted if self._state else False

    def check(self, nav: float, now: dt.datetime) -> KillSwitchCheck:
        if self._state is None:
            self.initialize(nav, now)
        state = self._state
        assert state is not None

        current_key = trading_day_key(now)
        if current_key != state.daily_key:
            state.daily_key = current_key
            state.daily_baseline_nav = nav  # new trading day: reset the daily drawdown baseline only

        state.high_water_mark = max(state.high_water_mark, nav)

        if state.halted:
            return KillSwitchCheck(halted=True, newly_tripped=False, reason=state.halt_reason)

        overall_drawdown_pct = (state.high_water_mark - nav) / state.high_water_mark * 100
        daily_drawdown_pct = (state.daily_baseline_nav - nav) / state.daily_baseline_nav * 100

        if overall_drawdown_pct >= self._limits.max_overall_drawdown_pct:
            return self._trip(
                f"overall drawdown {overall_drawdown_pct:.2f}% >= limit "
                f"{self._limits.max_overall_drawdown_pct:.2f}%"
            )
        if daily_drawdown_pct >= self._limits.max_daily_drawdown_pct:
            return self._trip(
                f"daily drawdown {daily_drawdown_pct:.2f}% >= limit "
                f"{self._limits.max_daily_drawdown_pct:.2f}%"
            )

        return KillSwitchCheck(halted=False, newly_tripped=False)

    def _trip(self, reason: str) -> KillSwitchCheck:
        assert self._state is not None
        self._state.halted = True
        self._state.halt_reason = reason
        return KillSwitchCheck(halted=True, newly_tripped=True, reason=reason)

    def reset(self, nav: float, now: dt.datetime) -> None:
        """Only ever called from an explicit manual action (dashboard 'resume trading' button) -
        never automatically, including at the next daily rollover."""
        self.initialize(nav, now)

    def manual_trip(self, nav: float, now: dt.datetime, reason: str = "manual emergency stop") -> KillSwitchCheck:
        """The dashboard's Emergency Stop button - halts trading immediately regardless of
        current drawdown, via the exact same halted/reset mechanism the automatic drawdown
        kill-switch uses (see app/api/routes_control.py), so it's a well-exercised code path
        rather than a second, untested way to stop the bot."""
        if self._state is None:
            self.initialize(nav, now)
        return self._trip(reason)
