def init(**kwargs):
    """Initialize the specified names in the specified databases.

    The general process is as follows:
      - Ensure the database in question exists
      - Ensure all tables exist in the database.
    """

    # TODO: Iterate through all engines in name set.
    database = kwargs.pop('database', False)
    if database and not database_exists(engine['default'].url):
        create_database(engine['default'].url, encoding='utf8')
        clear_cache()

    expression = lambda target, table: table.create(target)
    test = lambda target, table: table.exists(target)
    op(expression, test=test, primary='init', secondary='create', **kwargs)