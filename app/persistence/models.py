import datetime as dt
import enum

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class TradeDirection(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class ModelStatus(str, enum.Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    LIVE = "live"
    REJECTED = "rejected"
    RETIRED = "retired"


class Trade(Base):
    """A round-trip (or currently open) position, mirroring OANDA's own trade record."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    oanda_trade_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    instrument: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[TradeDirection] = mapped_column(Enum(TradeDirection))
    units: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[TradeStatus] = mapped_column(Enum(TradeStatus), default=TradeStatus.OPEN)
    strategy_name: Mapped[str] = mapped_column(String(64))
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )

    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="trades")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EquitySnapshot(Base):
    """Periodic snapshot of account NAV, used for the dashboard equity curve and drawdown calc."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    taken_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    nav: Mapped[float] = mapped_column(Float)
    balance: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float)


class ModelVersion(Base):
    """A trained (or rule-based) strategy artifact and its lifecycle status."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))  # "baseline" | "rl_candidate"
    # Nullable for backward compatibility with rows created before multi-instrument support -
    # those all predate this column and were all trained for EUR_USD (the only instrument that
    # existed then); repo/API code should treat a null instrument as "EUR_USD" for those old
    # rows rather than crashing on it. Every row written from here on always sets this.
    instrument: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    # Also nullable for the same backward-compat reason - old rows all used lookback=20 (the
    # only value that was ever passed), so treat a null lookback as 20 when loading one.
    lookback: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    training_data_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_data_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.CANDIDATE)
    validation_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    promoted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    trades: Mapped[list["Trade"]] = relationship(back_populates="model_version")


class KillSwitchEvent(Base):
    """Every trip and reset of the drawdown kill-switch, for audit."""

    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event: Mapped[str] = mapped_column(String(16))  # "tripped" | "reset"
    reason: Mapped[str] = mapped_column(String(255))
    nav_at_event: Mapped[float] = mapped_column(Float)
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
