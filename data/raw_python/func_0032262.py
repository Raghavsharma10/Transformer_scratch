def cache_key(*args, **kwargs):
    """
    Base method for computing the cache key with respect to the given
    arguments.
    """
    key = ""
    for arg in args:
        if callable(arg):
            key += ":%s" % repr(arg)
        else:
            key += ":%s" % str(arg)

    return key