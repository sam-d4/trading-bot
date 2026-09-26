"""Shared helper for loading the extra context (cross-pair candles, live carry rate) that
compute_features' richer feature set needs - used by both scripts/run_backtest.py and
scripts/run_training.py so the two don't duplicate (and risk drifting on) this logic."""

import structlog

from app.broker.oanda_client import OandaClient, OandaClientError
from app.config.settings import Settings
from app.data.features import related_instruments
from app.data.store import load_candles

log = structlog.get_logger(__name__)


def load_feature_context(settings: Settings, instrument: str, granularity: str) -> tuple[dict, float]:
    """Returns (related_candles, carry_differential) for compute_features. Missing cross-pair
    cache files are skipped with a warning rather than failing outright - compute_features fills
    a missing related instrument's column with 0, which is a degraded-but-safe fallback, not a
    silent correctness bug (0 momentum reads as "no signal from that pair", not "predicted flat
    market")."""
    related = {}
    for inst in related_instruments(instrument):
        try:
            related[inst] = load_candles(inst, granularity)
        except FileNotFoundError:
            log.warning("cross_pair_data_missing", instrument=inst, granularity=granularity)

    carry_differential = 0.0
    if settings.oanda_api_token and settings.oanda_account_id:
        try:
            client = OandaClient(settings)
            rates = client.get_financing_rates([instrument])
            carry_differential = rates.get(instrument, 0.0)
        except OandaClientError as exc:
            log.warning("financing_rate_fetch_failed", instrument=instrument, error=str(exc))

    return related, carry_differential
