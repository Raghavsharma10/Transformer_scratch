def getfieldcomm(bunchdt, data, commdct, idfobject, fieldname):
    """get the idd comment for the field"""
    key = idfobject.obj[0].upper()
    keyi = data.dtls.index(key)
    fieldi = idfobject.objls.index(fieldname)
    thiscommdct = commdct[keyi][fieldi]
    return thiscommdct