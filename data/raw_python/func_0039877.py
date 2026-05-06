def _unary(space,const,name):
    """
    Reduce the domain of variable name to be node-consistent with this
    constraint, i.e. remove those values for the variable that are not
    consistent with the constraint.

    returns True if the domain of name was modified
    """
    if not name in const.vnames:
        return False
    if space.variables[name].discrete:
        values = const.domains[name]
    else:
        values = const.domains[name]

    space.domains[name] = space.domains[name].intersection(values)
    return True