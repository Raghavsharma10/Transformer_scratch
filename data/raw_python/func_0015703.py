def find_library(name, cached=True, internal=True):
    """
        cached: Return a new instance
        internal: return a shared instance that's not the ctypes cached one
    """

    # a new one
    if not cached:
        return cdll.LoadLibrary(_so_mapping[name])

    # from the shared internal set or a new one
    if internal:
        if name not in _internal:
            _internal[name] = cdll.LoadLibrary(_so_mapping[name])
        return _internal[name]

    # a shared one
    return getattr(cdll, _so_mapping[name])