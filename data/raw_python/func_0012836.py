def getfieldidd_item(bch, fieldname, iddkey):
    """return an item from the fieldidd, given the iddkey
    will return and empty list if it does not have the iddkey
    or if the fieldname does not exist"""
    fieldidd = getfieldidd(bch, fieldname)
    try:
        return fieldidd[iddkey]
    except KeyError as e:
        return []