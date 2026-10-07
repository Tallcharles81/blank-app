import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

load_dotenv()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER', 'nfl_dfs')}"
        f":{os.getenv('PGPASSWORD', '')}"
        f"@{os.getenv('PGHOST', '127.0.0.1')}"
        f":{os.getenv('PGPORT', '5432')}"
        f"/{os.getenv('PGDATABASE', 'nfl_dfs')}"
    )
    return create_engine(url)


def run_migrations(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    schema_sql = SCHEMA_PATH.read_text()
    # Use the raw DBAPI connection instead of engine.execute(text(...)): SQLAlchemy
    # always passes a parameters dict to psycopg2, which then tries to %-interpolate
    # the literal "%" characters in schema.sql's RAISE EXCEPTION message and fails.
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(schema_sql)
        raw_conn.commit()
    finally:
        raw_conn.close()


if __name__ == "__main__":
    run_migrations()
    print("Migrations applied.")
