"""Deterministic position sizing - lives entirely in the risk layer, never in a strategy or the
RL model (see app/strategy/base.py's docstring). Two independent safeguards are applied: a
fixed-fractional risk formula sized off current volatility, then a hard leverage clamp that wins
regardless of what the formula computes, so a sizing-logic bug can't over-bet the account.

Currency conversion: the risk budget (risk_per_trade_pct of equity) is denominated in the
ACCOUNT's currency, but the stop distance and price are denominated in the traded instrument's
QUOTE currency - dividing one by the other without converting is only correct when those happen
to be the same currency (e.g. a USD account trading EUR_USD). Found as a real bug on this
project's actual account (GBP, trading EUR_USD/quote=USD): risk was silently running ~35% off
target at the GBP/USD rate prevailing when it was found, because no conversion was applied at
all. account_to_quote_rate fixes this generally - callers resolve it via
resolve_account_to_quote_rate below, using whatever live cross-rate is available.
"""

from dataclasses import dataclass

from app.risk.limits import RiskLimits


@dataclass(frozen=True)
class SizingResult:
    units: float
    capped_by_leverage: bool


def size_position(
    *,
    equity: float,
    price: float,
    atr: float,
    direction: int,  # +1 long, -1 short, 0 -> always returns 0 units
    limits: RiskLimits,
    atr_stop_multiplier: float = 2.0,
    account_to_quote_rate: float = 1.0,
) -> SizingResult:
    """account_to_quote_rate: the price of one unit of the account's currency, expressed in the
    traded instrument's quote currency (e.g. for a GBP account trading EUR_USD, this is the
    GBP_USD rate - "how many USD is 1 GBP worth"). Defaults to 1.0, correct only when the account
    currency already equals the quote currency; every other case needs the real rate passed in,
    or risk sizing is silently wrong by whatever that rate actually is.
    """
    if direction == 0 or equity <= 0 or price <= 0:
        return SizingResult(units=0.0, capped_by_leverage=False)

    equity_in_quote_currency = equity * account_to_quote_rate

    stop_distance = max(atr * atr_stop_multiplier, price * 0.0001)  # floor avoids div-by-~0 on dead ticks
    risk_amount = equity_in_quote_currency * (limits.risk_per_trade_pct / 100)
    raw_units = risk_amount / stop_distance

    max_units_from_leverage = (equity_in_quote_currency * limits.max_leverage) / price
    capped = raw_units > max_units_from_leverage
    units = min(raw_units, max_units_from_leverage)

    return SizingResult(units=direction * units, capped_by_leverage=capped)


def resolve_account_to_quote_rate(
    account_currency: str, quote_currency: str, rate_lookup: dict[str, float]
) -> float:
    """rate_lookup maps 'BASE_QUOTE' instrument names (OANDA's naming convention, e.g.
    'GBP_USD') to their current price. Returns the rate to convert 1 unit of account_currency
    into quote_currency for size_position's account_to_quote_rate.

    Raises ValueError if account_currency == quote_currency is False and neither the direct nor
    inverse pair is in rate_lookup - callers (app/core/engine.py) should catch this and fall back
    to rate=1.0 with a loud warning rather than let a missing cross-rate crash the live loop;
    that fallback is a known, logged approximation, not a silent one.
    """
    if account_currency == quote_currency:
        return 1.0

    def pair_rate(base: str, quote: str) -> float | None:
        if f"{base}_{quote}" in rate_lookup:
            return rate_lookup[f"{base}_{quote}"]
        if f"{quote}_{base}" in rate_lookup:
            return 1.0 / rate_lookup[f"{quote}_{base}"]
        return None

    direct = f"{account_currency}_{quote_currency}"
    inverse = f"{quote_currency}_{account_currency}"
    rate = pair_rate(account_currency, quote_currency)
    if rate is not None:
        return rate

    # No direct pair (e.g. GBP account, JPY-quoted USD_JPY: there is no GBP_JPY in the tracked
    # set) - cross through USD, which every tracked pair trades against.
    to_usd = pair_rate(account_currency, "USD")
    usd_to_quote = pair_rate("USD", quote_currency)
    if to_usd is not None and usd_to_quote is not None:
        return to_usd * usd_to_quote

    raise ValueError(
        f"no conversion rate available for {account_currency} -> {quote_currency} "
        f"(looked for {direct} and {inverse} in rate_lookup)"
    )
