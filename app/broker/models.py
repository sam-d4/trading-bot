from pydantic import BaseModel


class AccountSummary(BaseModel):
    account_id: str
    currency: str
    balance: float
    nav: float
    unrealized_pl: float
    margin_used: float
    margin_available: float
    open_trade_count: int
    open_position_count: int


class OpenTrade(BaseModel):
    trade_id: str
    instrument: str
    units: float  # positive = long, negative = short
    price: float
    unrealized_pl: float
    open_time: str


class Tick(BaseModel):
    instrument: str
    time: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class OrderResult(BaseModel):
    order_id: str | None
    trade_id: str | None
    fill_price: float | None
    status: str
    raw: dict
