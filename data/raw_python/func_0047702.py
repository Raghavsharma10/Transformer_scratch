def booleanise(b):
    """Normalise a 'stringified' Boolean to a proper Python Boolean.

    ElasticSearch has a habit of returning "true" and "false" in its 
    JSON responses when it should be returning `true` and `false`.  If 
    `b` looks like a stringified Boolean true, return True.  If `b` 
    looks like a stringified Boolean false, return False.

    If we don't know what `b` is supposed to represent, return it back 
    to the caller.

    """
    s = str(b)
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    return b