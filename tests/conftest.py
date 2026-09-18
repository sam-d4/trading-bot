import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
# See app/rl/__init__.py - avoids a matplotlib/macOS font-scan crash triggered by importing
# stable_baselines3 (pulled in by tests/test_rl_pipeline.py). Set here so it's in place before
# pytest collects any test module, regardless of import order.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence.models import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
