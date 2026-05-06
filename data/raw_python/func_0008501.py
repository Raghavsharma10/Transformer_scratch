def zip(*args, **kwargs):
    """ Returns a list of tuples, where the i-th tuple contains the i-th element 
        from each of the argument sequences or iterables (or default if too short).
    """
    args = [list(iterable) for iterable in args]
    n = max(map(len, args))
    v = kwargs.get("default", None)
    return _zip(*[i + [v] * (n - len(i)) for i in args])