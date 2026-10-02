import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "nfl_dfs"),
        user=os.getenv("PGUSER", "nfl_dfs"),
        password=os.getenv("PGPASSWORD", ""),
    )
