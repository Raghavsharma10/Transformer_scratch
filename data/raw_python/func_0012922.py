def group2commdct(commdct, glist):
    """add group info tocomdct"""
    for (gname, objname), commitem in zip(glist, commdct):
        commitem[0]['group'] = gname
        commitem[0]['idfobj'] = objname
    return commdct