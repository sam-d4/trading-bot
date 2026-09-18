"""Regression coverage for the Infinity/NaN JSON bug: Python's json.dumps happily emits the
bare tokens Infinity/-Infinity/NaN (a non-standard extension), which a real, spec-compliant
JSON.parse (every browser) rejects with a SyntaxError. A metrics dict with an infinite
profit_factor - any strategy with zero losing trades, which buy_and_hold hits constantly -
used to serialize "successfully" via repo.set_model_status and then fail to parse in the
dashboard silently, with the promotion card's comparison table just not rendering."""

import json
import math

from app.persistence import repo
from app.persistence.models import ModelStatus


def test_json_safe_replaces_non_finite_floats_with_none():
    data = {
        "candidate": {"sharpe": 1.5, "profit_factor": float("inf")},
        "comparison": {"buy_and_hold": {"profit_factor": float("inf"), "sharpe": -float("inf")}},
        "nan_value": float("nan"),
        "ordinary_list": [1.0, float("inf"), 2.0],
    }
    safe = repo._json_safe(data)

    assert safe["candidate"]["profit_factor"] is None
    assert safe["candidate"]["sharpe"] == 1.5  # ordinary finite floats pass through untouched
    assert safe["comparison"]["buy_and_hold"]["profit_factor"] is None
    assert safe["comparison"]["buy_and_hold"]["sharpe"] is None
    assert safe["nan_value"] is None
    assert safe["ordinary_list"] == [1.0, None, 2.0]

    # the whole point: the sanitized result must be valid per the JSON spec, not just to Python's
    # own lenient loads/dumps (which would happily round-trip the un-sanitized Infinity too).
    reserialized = json.dumps(safe, allow_nan=False)
    assert "Infinity" not in reserialized
    assert "NaN" not in reserialized


def test_set_model_status_sanitizes_infinite_metrics(db_session):
    mv = repo.create_model_version(db_session, name="test_model", kind="rl_candidate", instrument="EUR_USD", lookback=20)
    repo.set_model_status(
        db_session,
        mv.id,
        ModelStatus.VALIDATED,
        validation_metrics={"candidate": {"profit_factor": float("inf"), "sharpe": 1.2}},
    )
    db_session.refresh(mv)

    # must be strict-JSON parseable (mirrors what the browser's JSON.parse actually enforces),
    # not just parseable by Python's own lenient json.loads.
    parsed = json.loads(mv.validation_metrics_json, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    assert parsed["candidate"]["profit_factor"] is None
    assert parsed["candidate"]["sharpe"] == 1.2
