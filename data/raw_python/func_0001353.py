def connect(url=None, schema=None, sql_path=None, multiprocessing=False):
    """Open a new connection to postgres via psycopg2/sqlalchemy
    """
    if url is None:
        url = os.environ.get("DATABASE_URL")
    return Database(url, schema, sql_path=sql_path, multiprocessing=multiprocessing)