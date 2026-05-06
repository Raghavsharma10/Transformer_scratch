def branchlist2branches(data, commdct, branchlist):
    """get branches from the branchlist"""
    objkey = 'BranchList'.upper()
    theobjects = data.dt[objkey]
    fieldlists = []
    objnames = [obj[1] for obj in theobjects]
    for theobject in theobjects:
        fieldlists.append(list(range(2, len(theobject))))
    blists = extractfields(data, commdct, objkey, fieldlists)
    thebranches = [branches for name, branches in zip(objnames, blists)
                   if name == branchlist]
    return thebranches[0]