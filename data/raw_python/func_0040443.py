def quote_value(value):
    """return the value ready to be used as a value in a SQL string.

    For example you can safely do this:

        cursor.execute('select * from table where key = %s' % quote_value(val))

    and you don't have to worry about possible SQL injections.
    """
    adapted = adapt(value)
    if hasattr(adapted, 'getquoted'):
        adapted = adapted.getquoted()
    return adapted