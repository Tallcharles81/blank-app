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
