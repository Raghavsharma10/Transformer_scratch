def addthisbunch(bunchdt, data, commdct, thisbunch, theidf):
    """add a bunch to model.
    abunch usually comes from another idf file
    or it can be used to copy within the idf file"""
    key = thisbunch.key.upper()
    obj = copy.copy(thisbunch.obj)
    abunch = obj2bunch(data, commdct, obj)
    bunchdt[key].append(abunch)
    return abunch