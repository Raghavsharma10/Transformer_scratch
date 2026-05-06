def get_affected_records(spec=None, search_pattern=None):
    """Get list of affected records.

    :param spec: The record spec.
    :param search_pattern: The search pattern.
    :returns: An iterator to lazily find results.
    """
    # spec       pattern    query
    # ---------- ---------- -------
    # None       None       None
    # None       Y          Y
    # X          None       X
    # X          ''         X
    # X          Y          X OR Y

    if spec is None and search_pattern is None:
        raise StopIteration

    queries = []

    if spec is not None:
        queries.append(Q('match', **{'_oai.sets': spec}))

    if search_pattern:
        queries.append(query_string_parser(search_pattern=search_pattern))

    search = OAIServerSearch(
        index=current_app.config['OAISERVER_RECORD_INDEX'],
    ).query(Q('bool', should=queries))

    for result in search.scan():
        yield result.meta.id