def rename(idf, objkey, objname, newname):
    """rename all the refrences to this objname"""
    refnames = getrefnames(idf, objkey)
    for refname in refnames:
        objlists = getallobjlists(idf, refname)
        # [('OBJKEY', refname, fieldindexlist), ...]
        for refname in refnames:
        # TODO : there seems to be a duplication in this loop. Check.
        # refname appears in both loops
            for robjkey, refname, fieldindexlist in objlists:
                idfobjects = idf.idfobjects[robjkey]
                for idfobject in idfobjects:
                    for findex in fieldindexlist:  # for each field
                        if idfobject[idfobject.objls[findex]] == objname:
                            idfobject[idfobject.objls[findex]] = newname
    theobject = idf.getobject(objkey, objname)
    fieldname = [item for item in theobject.objls if item.endswith('Name')][0]
    theobject[fieldname] = newname
    return theobject