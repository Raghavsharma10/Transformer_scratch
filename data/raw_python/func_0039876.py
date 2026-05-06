def ac3(space):
    """
    AC-3 algorithm. This reduces the domains of the variables by
    propagating constraints to ensure arc consistency.

    :param Space space: The space to reduce
    """
    #determine arcs
    arcs = {}
    for name in space.variables:
        arcs[name] = set([])
    for const in space.constraints:
        for vname1,vname2 in product(const.vnames,const.vnames):
            if vname1 != vname2:
                #this is pessimistic, we assume that each constraint
                #pairwisely couples all variables it affects
                arcs[vname1].add(vname2)

    #enforce node consistency
    for vname in space.variables:
        for const in space.constraints:
            _unary(space,const,vname)

    #assemble work list
    worklist = set([])
    for v1 in space.variables:
        for v2 in space.variables:
            for const in space.constraints:
                if _binary(space,const,v1,v2):
                    for name in arcs[v1]:
                        worklist.add((v1,name))

    #work through work list
    while worklist:
        v1,v2 = worklist.pop()
        for const in space.constraints:
            if _binary(space,const,v1,v2):
                for vname in arcs[v1]:
                    worklist.add((v1,vname))