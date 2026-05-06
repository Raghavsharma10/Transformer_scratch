def copyidfintoidf(toidf, fromidf):
    """copy fromidf completely into toidf"""
    idfobjlst = getidfobjectlist(fromidf)
    for idfobj in idfobjlst:
        toidf.copyidfobject(idfobj)