def execute(query, data=None):
    """
    Execute an arbitrary SQL query given by query, returning any
    results as a list of OrderedDicts. A list of values can be supplied as an,
    additional argument, which will be substituted into question marks in the
    query.
    """
    connection = _State.connection()
    _State.new_transaction()

    if data is None:
        data = []

    result = connection.execute(query, data)

    _State.table = None
    _State.metadata = None
    try:
        del _State.table_pending
    except AttributeError:
        pass

    if not result.returns_rows:
        return {u'data': [], u'keys': []}

    return {u'data': result.fetchall(), u'keys': list(result.keys())}