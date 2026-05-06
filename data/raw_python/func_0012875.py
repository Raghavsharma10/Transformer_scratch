def embedgroupdata(extract_func, fname, debug):
    """insert group info into extracted idd"""
    
    astr = _readfname(fname)
    
    # fname is exhausted by the above read
    # reconstitute fname as a StringIO
    fname = StringIO(astr)

    try:
        astr = astr.decode('ISO-8859-2')
    except Exception as e:
        pass # for python 3
    glist = iddgroups.iddtxt2grouplist(astr)
    
    
    blocklst, commlst, commdct = extract_func(fname)
    # add group information to commlst and commdct
    # glist = getglist(fname)
    commlst = iddgroups.group2commlst(commlst, glist)
    commdct = iddgroups.group2commdct(commdct, glist)
    return blocklst, commlst, commdct