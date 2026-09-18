"""ModelVersion lifecycle management: candidate -> validated -> live/rejected. The only place
that should transition a model's status, so the promotion rules from the plan (manual
confirmation by default) live in one spot rather than being re-implemented per caller.
"""

import datetime as dt

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.core.events import Event, EventType, event_bus
from app.persistence import repo
from app.persistence.models import ModelStatus, ModelVersion
from app.rl.evaluate import ValidationOutcome


def register_candidate(
    session: Session,
    *,
    name: str,
    instrument: str,
    lookback: int,
    artifact_path: str,
    training_data_start: dt.datetime,
    training_data_end: dt.datetime,
) -> ModelVersion:
    return repo.create_model_version(
        session,
        name=name,
        kind="rl_candidate",
        instrument=instrument,
        lookback=lookback,
        artifact_path=artifact_path,
        training_data_start=training_data_start,
        training_data_end=training_data_end,
    )


def apply_validation_outcome(
    session: Session, settings: Settings, model_id: int, outcome: ValidationOutcome
) -> ModelVersion | None:
    """Applies the validation gate's verdict: REJECTED on failure (live model untouched);
    on a pass, either VALIDATED (queued for the dashboard's manual-confirmation promotion card -
    the default, per require_manual_confirmation_to_promote) or promoted straight to LIVE if
    the user has explicitly opted out of that safeguard."""
    if not outcome.passed:
        return repo.set_model_status(
            session,
            model_id,
            ModelStatus.REJECTED,
            validation_metrics={"candidate": outcome.candidate_metrics, "reasons": outcome.reasons},
        )

    validation_payload = {"candidate": outcome.candidate_metrics, "comparison": outcome.comparison_metrics}

    if settings.require_manual_confirmation_to_promote:
        mv = repo.set_model_status(session, model_id, ModelStatus.VALIDATED, validation_metrics=validation_payload)
        event_bus.publish(
            Event(EventType.MODEL_PROMOTION_PENDING, {"model_id": model_id, "metrics": outcome.candidate_metrics})
        )
    else:
        mv = repo.set_model_status(session, model_id, ModelStatus.LIVE, validation_metrics=validation_payload)
        event_bus.publish(
            Event(EventType.MODEL_PROMOTED, {"model_id": model_id, "metrics": outcome.candidate_metrics})
        )

    return mv
