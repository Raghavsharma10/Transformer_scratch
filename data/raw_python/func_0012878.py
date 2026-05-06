def readdatacommdct(idfname, iddfile='Energy+.idd', commdct=None):
    """read the idf file"""
    if not commdct:
        block, commlst, commdct, idd_index = parse_idd.extractidddata(iddfile)
        theidd = eplusdata.Idd(block, 2)
    else:
        theidd = iddfile
    data = eplusdata.Eplusdata(theidd, idfname)
    return data, commdct, idd_index