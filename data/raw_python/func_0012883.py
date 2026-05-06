def branch_inlet_outlet(data, commdct, branchname):
    """return the inlet and outlet of a branch"""
    objkey = 'Branch'.upper()
    theobjects = data.dt[objkey]
    theobject = [obj for obj in theobjects if obj[1] == branchname]
    theobject = theobject[0]
    inletindex = 6
    outletindex = len(theobject) - 2
    return [theobject[inletindex], theobject[outletindex]]