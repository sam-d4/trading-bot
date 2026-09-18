from app.persistence.models import ModelStatus, ModelVersion, Trade, TradeDirection, TradeStatus


def test_trade_round_trip(db_session):
    trade = Trade(
        oanda_trade_id="1",
        instrument="EUR_USD",
        direction=TradeDirection.LONG,
        units=1000,
        entry_price=1.1000,
        opened_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        status=TradeStatus.OPEN,
        strategy_name="test",
    )
    db_session.add(trade)
    db_session.commit()

    fetched = db_session.query(Trade).one()
    assert fetched.instrument == "EUR_USD"
    assert fetched.direction == TradeDirection.LONG
    assert fetched.status == TradeStatus.OPEN


def test_model_version_default_status(db_session):
    mv = ModelVersion(name="baseline-trend", kind="baseline")
    db_session.add(mv)
    db_session.commit()

    fetched = db_session.query(ModelVersion).one()
    assert fetched.status == ModelStatus.CANDIDATE
