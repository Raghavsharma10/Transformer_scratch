def makebranchcomponents(data, commdct, anode="epnode"):
    """return the edges jointing the components of a branch"""
    alledges = []

    objkey = 'BRANCH'
    cnamefield = "Component %s Name"
    inletfield = "Component %s Inlet Node Name"
    outletfield = "Component %s Outlet Node Name"

    numobjects = len(data.dt[objkey])
    cnamefields = loops.repeatingfields(data, commdct, objkey, cnamefield)
    inletfields = loops.repeatingfields(data, commdct, objkey, inletfield)
    outletfields = loops.repeatingfields(data, commdct, objkey, outletfield)

    inlts = loops.extractfields(data, commdct, 
        objkey, [inletfields] * numobjects)
    cmps = loops.extractfields(data, commdct, 
        objkey, [cnamefields] * numobjects)
    otlts = loops.extractfields(data, commdct, 
        objkey, [outletfields] * numobjects)

    zipped = list(zip(inlts, cmps, otlts))
    tzipped = [transpose2d(item) for item in zipped]
    for i in range(len(data.dt[objkey])):
        tt = tzipped[i]
        # branchname = data.dt[objkey][i][1]
        edges = []
        for t0 in tt:
            edges = edges + [((t0[0], anode), t0[1]), (t0[1], (t0[2], anode))]
        alledges = alledges + edges
    return alledges