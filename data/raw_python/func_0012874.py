def make_idd_index(extract_func, fname, debug):
    """generate the iddindex"""
    astr = _readfname(fname)

    # fname is exhausted by the above read
    # reconstitute fname as a StringIO
    fname = StringIO(astr)

    # glist = iddgroups.iddtxt2grouplist(astr.decode('ISO-8859-2'))
    
    
    blocklst, commlst, commdct = extract_func(fname)
    
    name2refs = iddindex.makename2refdct(commdct)
    ref2namesdct = iddindex.makeref2namesdct(name2refs)
    idd_index = dict(name2refs=name2refs, ref2names=ref2namesdct)
    commdct = iddindex.ref2names2commdct(ref2namesdct, commdct)
    
    return blocklst, commlst, commdct, idd_index