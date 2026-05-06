def _copy(query_dict):
    """
    Return a mutable copy of `query_dict`. This is a workaround to
    Django bug #13572, which prevents QueryDict.copy from working.
    """

    memo = { }

    result = query_dict.__class__('',
        encoding=query_dict.encoding,
        mutable=True)

    memo[id(query_dict)] = result

    for key, value in dict.items(query_dict):
        dict.__setitem__(result,
            copy.deepcopy(key, memo),
            copy.deepcopy(value, memo))

    return result