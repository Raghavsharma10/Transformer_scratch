def get_suggestions(idx, unresolved):
    """
    Returns suggestions
    """
    result = {}
    for u, lines in unresolved.items():
        paths = idx.get(u)
        if paths:
            result[u] = {'paths': paths, 'lineno': lines}
    return result