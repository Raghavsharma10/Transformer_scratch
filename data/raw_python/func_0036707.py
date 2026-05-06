def append_condition(statement, condition, key, value):
    """
    >>> list = []
    >>> append_condition(list, 'match', 'name', 'Jack')
    >>> list
    [{'match': {'name': 'Jack'}}]
    >>> dict = {}
    >>> append_condition(dict, 'match', 'name', 'Marry')
    >>> dict
    {'match': {'name': 'Marry'}}
    """
    if isinstance(statement, list):
        statement.append({condition: {key: value}})
    if isinstance(statement, dict):
        statement[condition] = {key: value}