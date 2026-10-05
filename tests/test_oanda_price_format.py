from app.broker.oanda_client import format_price, format_stop_distance


def test_jpy_pairs_use_three_decimals_and_others_five():
    assert format_stop_distance("USD_JPY", 0.4137) == "0.414"
    assert format_stop_distance("EUR_USD", 0.003012345) == "0.00301"
    assert format_price("USD_JPY", 157.23456) == "157.235"


def test_a_tiny_distance_is_floored_at_one_tick_never_zero():
    assert format_stop_distance("EUR_USD", 0.0000001) == "0.00001"
    assert format_stop_distance("USD_JPY", 0.0001) == "0.001"
