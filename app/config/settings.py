from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OANDA
    oanda_api_token: str = ""
    oanda_account_id: str = ""
    oanda_environment: Literal["practice", "live"] = "practice"

    # Instruments
    primary_instrument: str = "EUR_USD"  # the default/first instrument - used where a single one is needed
    # Comma-separated, not a native list, so a plain string in .env is enough (pydantic-settings'
    # env-var list parsing wants JSON syntax, which is an awkward fit for a human-edited .env).
    traded_instruments_csv: str = "EUR_USD,GBP_USD,USD_JPY,AUD_USD"

    # Live engine timing (seconds)
    live_bar_seconds: int = 300
    kill_switch_poll_seconds: int = 30
    reconciliation_interval_seconds: int = 60
    live_model_poll_seconds: int = 120
    rl_lookback: int = 20

    # Risk limits
    max_daily_drawdown_pct: float = 3.0
    max_overall_drawdown_pct: float = 10.0
    risk_per_trade_pct: float = 1.0
    max_leverage: float = 10.0
    flatten_on_kill_switch: bool = True

    # Self-improvement
    require_manual_confirmation_to_promote: bool = True

    # Storage
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'trading_bot.db'}"
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    dashboard_basic_auth_user: str = ""
    dashboard_basic_auth_pass: str = ""

    # Alerting
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""

    @property
    def oanda_rest_host(self) -> str:
        return (
            "https://api-fxpractice.oanda.com"
            if self.oanda_environment == "practice"
            else "https://api-fxtrade.oanda.com"
        )

    @property
    def oanda_stream_host(self) -> str:
        return (
            "https://stream-fxpractice.oanda.com"
            if self.oanda_environment == "practice"
            else "https://stream-fxtrade.oanda.com"
        )

    @property
    def is_live(self) -> bool:
        return self.oanda_environment == "live"

    @property
    def traded_instruments(self) -> list[str]:
        return [i.strip() for i in self.traded_instruments_csv.split(",") if i.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
