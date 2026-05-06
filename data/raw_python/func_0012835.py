def getfieldidd(bch, fieldname):
    """get the idd dict for this field
    Will return {} if the fieldname does not exist"""
    # print(bch)
    try:
        fieldindex = bch.objls.index(fieldname)
    except ValueError as e:
        return {}  # the fieldname does not exist
                    # so there is no idd
    fieldidd = bch.objidd[fieldindex]
    return fieldidd