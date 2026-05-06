def checkrange(bch, fieldname):
    """throw exception if the out of range"""
    fieldvalue = bch[fieldname]
    therange = bch.getrange(fieldname)
    if therange['maximum'] != None:
        if fieldvalue > therange['maximum']:
            astr = "Value %s is not less or equal to the 'maximum' of %s"
            astr = astr % (fieldvalue, therange['maximum'])
            raise RangeError(astr)
    if therange['minimum'] != None:
        if fieldvalue < therange['minimum']:
            astr = "Value %s is not greater or equal to the 'minimum' of %s"
            astr = astr % (fieldvalue, therange['minimum'])
            raise RangeError(astr)
    if therange['maximum<'] != None:
        if fieldvalue >= therange['maximum<']:
            astr = "Value %s is not less than the 'maximum<' of %s"
            astr = astr % (fieldvalue, therange['maximum<'])
            raise RangeError(astr)
    if  therange['minimum>'] != None:
        if fieldvalue <= therange['minimum>']:
            astr = "Value %s is not greater than the 'minimum>' of %s"
            astr = astr % (fieldvalue, therange['minimum>'])
            raise RangeError(astr)
    return fieldvalue
    """get the idd dict for this field
    Will return {} if the fieldname does not exist"""