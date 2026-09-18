"""Alerting - Telegram primary, SMTP email fallback. Fires on: kill-switch trip, stream
disconnects, retrain failures, and model-pending-promotion. A crashed bot process can't alert
about its own death, so this is deliberately supplemented by an external uptime monitor
(healthchecks.io or similar) hitting /healthz, configured outside this codebase.
"""

import smtplib
from email.message import EmailMessage

import httpx
import structlog

from app.config.settings import Settings

log = structlog.get_logger(__name__)


class AlertSender:
    def __init__(self, settings: Settings):
        self._settings = settings

    def send(self, subject: str, message: str) -> None:
        sent = False
        if self._settings.telegram_bot_token and self._settings.telegram_chat_id:
            sent = self._send_telegram(f"*{subject}*\n{message}") or sent
        if self._settings.smtp_host and self._settings.alert_email_to:
            sent = self._send_email(subject, message) or sent
        if not sent:
            log.warning("alert_not_delivered_no_channel_configured", subject=subject)

    def _send_telegram(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self._settings.telegram_bot_token}/sendMessage"
        try:
            resp = httpx.post(
                url,
                json={"chat_id": self._settings.telegram_chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.error("telegram_alert_failed", error=str(exc))
            return False

    def _send_email(self, subject: str, message: str) -> bool:
        msg = EmailMessage()
        msg["Subject"] = f"[trading-bot] {subject}"
        msg["From"] = self._settings.smtp_user or "trading-bot@localhost"
        msg["To"] = self._settings.alert_email_to
        msg.set_content(message)
        try:
            with smtplib.SMTP(self._settings.smtp_host, self._settings.smtp_port, timeout=10) as smtp:
                smtp.starttls()
                if self._settings.smtp_user:
                    smtp.login(self._settings.smtp_user, self._settings.smtp_password)
                smtp.send_message(msg)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            log.error("email_alert_failed", error=str(exc))
            return False
