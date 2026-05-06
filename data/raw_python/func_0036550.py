def check_validity_for_dict(keys, dict):
    """
    >>> dict = {'a': 0, 'b': 1, 'c': 2}
    >>> keys = ['a', 'd', 'e']
    >>> check_validity_for_dict(keys, dict) == False
    True
    >>> keys = ['a', 'b', 'c']
    >>> check_validity_for_dict(keys, dict) == False
    False
    """
    for key in keys:
        if key not in dict or dict[key] is '' or dict[key] is None:
            return False
    return True