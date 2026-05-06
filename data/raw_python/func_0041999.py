def flush(**kwargs):
    """Flush the specified names from the specified databases.

    This can be highly destructive as it destroys all data.
    """

    expression = lambda target, table: target.execute(table.delete())
    test = lambda target, table: not table.exists(target)
    op(expression, reversed(metadata.sorted_tables), test=test,
       primary='flush', secondary='flush', **kwargs)