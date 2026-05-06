def auth_db_connect(db_path):
    """ An SQLite database is used to store authentication transient data,
    this is tokens, strings of random data which are signed by the client,
    and session_tokens which identify authenticated users """

    def dict_factory(cursor, row): return {col[0] : row[idx] for idx,col in enumerate(cursor.description)}
    conn = db.connect(db_path)
    conn.row_factory = dict_factory
    if not auth_db_connect.init:
        conn.execute('create table if not exists tokens (expires int, token text, ip text)')
        conn.execute('create table if not exists session_tokens (expires int, token text, ip text, username text)')
        auth_db_connect.init = True
    return conn