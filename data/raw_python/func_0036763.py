def upsert(db, table, key_cols, update_dict):
    """Fabled upsert for SQLiteDB.

    Perform an upsert based on primary key.

    :param SQLiteDB db: database
    :param str table: table to upsert into
    :param str key_cols: name of key columns
    :param dict update_dict: key-value pairs to upsert

    """
    with db:
        cur = db.cursor()
        cur.execute(
            'UPDATE {} SET {} WHERE {}'.format(
                table,
                ','.join(_sqlpformat(col) for col in update_dict.keys()),
                ' AND '.join(_sqlpformat(col) for col in key_cols),
            ),
            update_dict,
        )
        if db.changes() == 0:
            keys, values = zip(*update_dict.items())
            cur.execute(
                'INSERT INTO {} ({}) VALUES ({})'.format(
                    table,
                    ','.join(keys),
                    ','.join('?' for _ in values)),
                values)