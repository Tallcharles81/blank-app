import pytest

from db.migrate import get_engine


@pytest.fixture(scope="session")
def engine():
    return get_engine()
