def _preprocess_sqlite_index(asql_query, library, backend, connection):
    """ Creates materialized view for each indexed partition found in the query.

    Args:
        asql_query (str): asql query
        library (ambry.Library):
        backend (SQLiteBackend):
        connection (apsw.Connection):

    Returns:
        str: converted asql if it contains index query. If not, returns asql_query as is.
    """

    new_query = None

    if asql_query.strip().lower().startswith('index'):

        logger.debug(
            '_preprocess_index: create index query found.\n    asql query: {}'
            .format(asql_query))

        index = parse_index(asql_query)
        partition = library.partition(index.source)
        table = backend.install(connection, partition, materialize=True)
        index_name = '{}_{}_ind'.format(partition.vid, '_'.join(index.columns))
        new_query = 'CREATE INDEX IF NOT EXISTS {index} ON {table} ({columns});'.format(
            index=index_name, table=table, columns=','.join(index.columns))

    logger.debug(
        '_preprocess_index: preprocess finished.\n    asql query: {}\n    new query: {}'
        .format(asql_query, new_query))

    return new_query or asql_query