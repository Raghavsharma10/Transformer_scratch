def _get_table_names(statement):
    """ Returns table names found in the query.

    NOTE. This routine would use the sqlparse parse tree, but vnames don't parse very well.

    Args:
        statement (sqlparse.sql.Statement): parsed by sqlparse sql statement.

    Returns:
        list of str
    """

    parts = statement.to_unicode().split()

    tables = set()

    for i, token in enumerate(parts):
        if token.lower() == 'from' or token.lower().endswith('join'):
            tables.add(parts[i + 1].rstrip(';'))

    return list(tables)