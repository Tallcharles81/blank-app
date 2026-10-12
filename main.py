from db.connection import get_connection


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
        print(f"Database connection OK: {version}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
