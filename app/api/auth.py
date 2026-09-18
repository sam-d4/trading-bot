"""Basic auth guard for the dashboard/control API - this bot places real orders, so it should
never be reachable unauthenticated once exposed beyond localhost. A no-op when
DASHBOARD_BASIC_AUTH_USER isn't set, so local development stays frictionless; set it before
exposing the dashboard on the VPS."""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config.settings import get_settings

_security = HTTPBasic(auto_error=False)


def require_basic_auth(credentials: HTTPBasicCredentials | None = Depends(_security)) -> None:
    settings = get_settings()
    if not settings.dashboard_basic_auth_user:
        return  # auth not configured - allowed for local dev only

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )
    if credentials is None:
        raise unauthorized

    user_ok = secrets.compare_digest(credentials.username, settings.dashboard_basic_auth_user)
    pass_ok = secrets.compare_digest(credentials.password, settings.dashboard_basic_auth_pass)
    if not (user_ok and pass_ok):
        raise unauthorized
