def select(query, data=None):
    """
    Perform a sql select statement with the given query (without 'select') and
    return any results as a list of OrderedDicts.
    """
    connection = _State.connection()
    _State.new_transaction()
    if data is None:
        data = []

    result = connection.execute('select ' + query, data)

    rows = []
    for row in result:
        rows.append(dict(list(row.items())))

    return rows