import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import routes_control, routes_dashboard, ws
from app.api.auth import require_basic_auth
from app.broker.oanda_client import OandaClient
from app.config.settings import get_settings
from app.core.engine import TradingEngine
from app.execution.order_manager import OrderManager
from app.notifications.alerts import AlertSender
from app.persistence import repo
from app.persistence.db import SessionLocal, init_db
from app.risk.limits import RiskLimits
from app.risk.manager import RiskManager
from app.strategy.baselines import TrendFollowingStrategy

log = structlog.get_logger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    log.info("startup", environment=settings.oanda_environment, instrument=settings.primary_instrument)

    limits = RiskLimits.from_settings(settings)
    app.state.settings = settings
    app.state.risk_manager = RiskManager(limits)
    app.state.oanda_client = None
    app.state.engine_task = None

    if settings.oanda_api_token and settings.oanda_account_id:
        client = OandaClient(settings)
        order_manager = OrderManager(client, app.state.risk_manager, limits)
        alerts = AlertSender(settings)

        # One strategy per traded instrument, independently hot-swappable - starts each on the
        # baseline unless that instrument already has a promoted live RL model from a previous
        # run (survives restarts, since it's read from the DB, not just engine memory).
        strategies = {}
        model_version_ids = {}
        with SessionLocal() as session:
            live_by_instrument = repo.current_live_models_by_instrument(session, settings.traded_instruments)

        for instrument in settings.traded_instruments:
            live_mv = live_by_instrument.get(instrument)
            if live_mv is not None and live_mv.artifact_path:
                try:
                    from app.data.features import feature_columns_for
                    from app.strategy.rl_policy import RLPolicyStrategy

                    strategies[instrument] = RLPolicyStrategy.load(
                        live_mv.artifact_path,
                        feature_columns_for(instrument),
                        lookback=live_mv.lookback or settings.rl_lookback,
                        model_version_name=live_mv.name,
                    )
                    model_version_ids[instrument] = live_mv.id
                    log.info("startup_loaded_live_model", instrument=instrument, name=live_mv.name)
                    continue
                except Exception as exc:  # noqa: BLE001 - a bad/missing artifact must not block startup
                    log.error("startup_live_model_load_failed_falling_back_to_baseline", instrument=instrument, error=str(exc))
            strategies[instrument] = TrendFollowingStrategy()
            model_version_ids[instrument] = None

        engine = TradingEngine(
            settings, client, strategies, app.state.risk_manager, order_manager, alerts, model_version_ids
        )

        app.state.oanda_client = client
        app.state.order_manager = order_manager
        app.state.engine = engine
        app.state.engine_task = asyncio.create_task(engine.run())
        log.info("engine_task_started", environment=settings.oanda_environment)

        # RL self-improvement loop - imported lazily so the base app (dashboard, trading engine
        # on baseline strategies) can still boot even if the optional [rl] extra (torch,
        # stable-baselines3) isn't installed. Needs cached historical data
        # (scripts/fetch_historical_data.py) to actually produce candidates; a no-op each cycle
        # until that exists.
        try:
            from app.rl.scheduler import start_retrain_scheduler

            app.state.retrain_scheduler = start_retrain_scheduler(settings, alerts)
        except ImportError as exc:
            log.warning("rl_extras_not_installed_self_improvement_disabled", error=str(exc))
            app.state.retrain_scheduler = None
    else:
        log.warning("oanda_credentials_missing_engine_not_started")
        app.state.retrain_scheduler = None

    yield

    if app.state.engine_task is not None:
        app.state.engine_task.cancel()
    if app.state.retrain_scheduler is not None:
        app.state.retrain_scheduler.shutdown(wait=False)


app = FastAPI(title="Trading Bot", lifespan=lifespan)

app.include_router(routes_dashboard.router, prefix="/api", dependencies=[Depends(require_basic_auth)])
app.include_router(routes_control.router, prefix="/api", dependencies=[Depends(require_basic_auth)])
app.include_router(ws.router)


@app.get("/healthz")
def healthz() -> dict:
    """Polled by an external uptime monitor - a crashed process can't alert about its own death."""
    return {"status": "ok"}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
