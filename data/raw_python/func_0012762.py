def isfieldvalue(bunchdt, data, commdct, idfobj, fieldname, value, places=7):
    """test if idfobj.field == value"""
    # do a quick type check
    # if type(idfobj[fieldname]) != type(value):
    # return False # takes care of autocalculate and real
    # check float
    thiscommdct = getfieldcomm(bunchdt, data, commdct, idfobj, fieldname)
    if 'type' in thiscommdct:
        if thiscommdct['type'][0] in ('real', 'integer'):
            # test for autocalculate
            try:
                if idfobj[fieldname].upper() == 'AUTOCALCULATE':
                    if value.upper() == 'AUTOCALCULATE':
                        return True
            except AttributeError:
                pass
            return almostequal(float(idfobj[fieldname]), float(value), places, False)
    # check retaincase
    if is_retaincase(bunchdt, data, commdct, idfobj, fieldname):
        return idfobj[fieldname] == value
    else:
        return idfobj[fieldname].upper() == value.upper()