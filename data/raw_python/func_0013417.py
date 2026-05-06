def iterativeFetch(query, batchSize=default_batch_size):
    """
    Returns rows of a sql fetch query on demand
    """
    while True:
        rows = query.fetchmany(batchSize)
        if not rows:
            break
        rowDicts = sqliteRowsToDicts(rows)
        for rowDict in rowDicts:
            yield rowDict