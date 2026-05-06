def parse_view(query):
    """ Parses asql query to view object.

    Args:
        query (str): asql query

    Returns:
        View instance: parsed view.
    """

    try:
        idx = query.lower().index('where')
        query = query[:idx]
    except ValueError:
        pass

    if not query.endswith(';'):
        query = query.strip()
        query += ';'

    result = _view_stmt.parseString(query)

    return View(result)