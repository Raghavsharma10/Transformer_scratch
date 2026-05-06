def clear(**kwargs):
    """Clear the specified names from the specified databases.

    This can be highly destructive as it destroys tables and when all names
    are removed from a database, the database itself.
    """

    database = kwargs.pop('database', False)
    expression = lambda target, table: table.drop(target)
    test = lambda x, tab: not database_exists(x.url) or not tab.exists(x)

    # TODO: Iterate through all engines in name set.
    if database and database_exists(engine['default'].url):
        drop_database(engine['default'].url)
        clear_cache()

    op(expression, reversed(metadata.sorted_tables), test=test,
       primary='clear', secondary='drop', **kwargs)