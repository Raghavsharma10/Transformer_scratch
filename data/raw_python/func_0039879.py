def solve(space,method='backtrack',ordering=None):
    """
    Generator for all solutions.


    :param str method: the solution method to employ
    :param ordering: an optional parameter ordering
    :type ordering: sequence of parameter names

    Methods:

    :"backtrack": simple chronological backtracking
    :"ac-lookahead": full lookahead
    """
    if ordering is None:
        ordering = list(space.variables.keys())

    if not space.is_discrete():
        raise ValueError("Can not backtrack on non-discrete space")
    if method=='backtrack':
        for label in _backtrack(space,{},ordering):
            yield label
    elif method=='ac-lookahead':
        for label in _lookahead(space,{},ordering):
            yield label
    else:
        raise ValueError("Unknown solution method: %s" % method)