def is_retaincase(bunchdt, data, commdct, idfobject, fieldname):
    """test if case has to be retained for that field"""
    thiscommdct = getfieldcomm(bunchdt, data, commdct, idfobject, fieldname)
    return 'retaincase' in thiscommdct