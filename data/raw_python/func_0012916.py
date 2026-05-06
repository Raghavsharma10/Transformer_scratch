def autosize_fieldname(idfobject):
    """return autsizeable field names in idfobject"""
    # undocumented stuff in this code
    return [fname for (fname, dct) in zip(idfobject.objls,
                                          idfobject['objidd'])
            if 'autosizable' in dct]