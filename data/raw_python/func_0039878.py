def _binary(space,const,name1,name2):
    """
    reduce the domain of variable name1 to be two-consistent (arc-consistent)
    with this constraint, i.e. remove those values for the variable name1,
    for which no values for name2 exist such that this pair is consistent
    with the constraint

    returns True if the domain of name1 was modified
    """
    if not (name1 in const.vnames and name2 in const.vnames):
        return False
    remove = set([])
    for v1 in space.domains[name1].iter_members():
        for v2 in space.domains[name2].iter_members():
            if const.consistent({name1 : v1, name2 : v2}):
                break
        else:
            remove.add(v1)

    if len(remove) > 0:
        if space.variables[name1].discrete:
            remove = DiscreteSet(remove)
        else:
            remove = IntervalSet.from_values(remove)

        space.domains[name1] = space.domains[name1].difference(remove)
        return True
    else:
        return False