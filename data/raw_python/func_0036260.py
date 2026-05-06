def filtered(path, filter):
    '''Check if a path is removed by a filter.

    Check if a path is in the provided set of paths, @ref filter. If
    none of the paths in filter begin with @ref path, then True is
    returned to indicate that the path is filtered out. If @ref path is
    longer than the filter, and starts with the filter, it is
    considered unfiltered (all paths below a filter are unfiltered).

    An empty filter ([]) is treated as not filtering any.

    '''
    if not filter:
        return False
    for p in filter:
        if len(path) > len(p):
            if path[:len(p)] == p:
                return False
        else:
            if p[:len(path)] == path:
                return False
    return True