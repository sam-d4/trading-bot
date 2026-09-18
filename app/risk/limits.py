from dataclasses import dataclass

from app.config.settings import Settings


@dataclass(frozen=True)
class RiskLimits:
    max_daily_drawdown_pct: float
    max_overall_drawdown_pct: float
    risk_per_trade_pct: float
    max_leverage: float
    flatten_on_kill_switch: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "RiskLimits":
        # risk_per_trade_pct is scaled down by how many pairs are being traded concurrently, so
        # aggregate worst-case exposure (all pairs stopped out the same day) stays roughly where
        # it was for one pair, rather than multiplying with every pair added. Trading 4 pairs at
        # the configured risk_per_trade_pct each, unscaled, could realize a same-day loss well
        # past max_daily_drawdown_pct before the kill-switch even gets a chance to halt further
        # trades between bars. Override via risk_per_trade_pct directly if you deliberately want
        # full risk stacked on every pair instead.
        num_instruments = max(len(settings.traded_instruments), 1)
        return cls(
            max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
            max_overall_drawdown_pct=settings.max_overall_drawdown_pct,
            risk_per_trade_pct=settings.risk_per_trade_pct / num_instruments,
            max_leverage=settings.max_leverage,
            flatten_on_kill_switch=settings.flatten_on_kill_switch,
        )
