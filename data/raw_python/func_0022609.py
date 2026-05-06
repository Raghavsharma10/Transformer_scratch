def search(query, data, replacements=None):
    """Yield objects from 'data' that match the 'query'."""
    query = q.Query(query, params=replacements)
    for entry in data:
        if solve.solve(query, entry).value:
            yield entry