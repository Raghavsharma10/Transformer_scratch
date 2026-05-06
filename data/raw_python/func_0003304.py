def ensure_keys(walk, *keys):
    """
    Use walk to try to retrieve all keys
    """
    all_retrieved = True
    for k in keys:
        try:
            walk(k)
        except WalkKeyNotRetrieved:
            all_retrieved = False
    return all_retrieved