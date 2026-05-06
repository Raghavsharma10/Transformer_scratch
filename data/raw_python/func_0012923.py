def commdct2grouplist(gcommdct):
    """extract embedded group data from commdct.
    return gdict -> {g1:[obj1, obj2, obj3], g2:[obj4, ..]}"""
    gdict = {}
    for objidd in gcommdct:
        group = objidd[0]['group']
        objname = objidd[0]['idfobj']
        if group in gdict:
            gdict[group].append(objname)
        else:
            gdict[group] = [objname, ]
    return gdict