def drop_db(url):
    """Drop specified database
    """
    parsed_url = urlparse(url)
    db_name = parsed_url.path
    db_name = db_name.strip("/")
    db = connect("postgresql://" + parsed_url.netloc)
    # check that db exists
    q = """SELECT 1 as exists
           FROM pg_database
           WHERE datname = '{db_name}'""".format(
        db_name=db_name
    )
    if db.query(q).fetchone():
        # DROP DATABASE must be run outside of a transaction
        conn = db.engine.connect()
        conn.execute("commit")
        conn.execute("DROP DATABASE " + db_name)
        conn.close()