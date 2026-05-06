def normalize_select(query):
    """
    If the query contains the :select operator, we enforce :sys properties.
    The SDK requires sys.type to function properly, but as other of our
    SDKs require more parts of the :sys properties, we decided that every
    SDK should include the complete :sys block to provide consistency
    accross our SDKs.
    """

    if 'select' not in query:
        return

    if isinstance(
            query['select'],
            str_type()):
        query['select'] = [s.strip() for s in query['select'].split(',')]

    query['select'] = [s for s
                       in query['select']
                       if not s.startswith('sys.')]

    if 'sys' not in query['select']:
        query['select'].append('sys')