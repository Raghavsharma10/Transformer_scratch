def getedges(fname, iddfile):
    """return the edges of the idf file fname"""
    data, commdct, _idd_index = readidf.readdatacommdct(fname, iddfile=iddfile)
    edges = makeairplantloop(data, commdct)
    return edges