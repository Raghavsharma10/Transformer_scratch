def group2commlst(commlst, glist):
    """add group info to commlst"""
    for (gname, objname), commitem in zip(glist, commlst):
        newitem1 = "group %s" % (gname, )
        newitem2 = "idfobj %s" % (objname, )
        commitem[0].insert(0, newitem1)
        commitem[0].insert(1, newitem2)
    return commlst