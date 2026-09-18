# --- frontend build stage ---
FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- backend runtime ---
FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY migrations/ ./migrations/
COPY alembic.ini ./

RUN pip install --no-cache-dir ".[rl]"

# stable-baselines3 pulls in matplotlib purely for optional plot logging we never use - this
# skips its system font scan (see app/rl/__init__.py for the full story).
ENV MPL_IGNORE_SYSTEM_FONTS=1

COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
