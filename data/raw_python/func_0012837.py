def isequal(bch, fieldname, value, places=7):
    """return True if the field is equal to value"""
    def equalalphanumeric(bch, fieldname, value):
        if bch.get_retaincase(fieldname):
            return bch[fieldname] == value
        else:
            return bch[fieldname].upper() == value.upper()

    fieldidd = bch.getfieldidd(fieldname)
    try:
        ftype = fieldidd['type'][0]
        if ftype in ['real', 'integer']:
            return almostequal(bch[fieldname], float(value), places=places)
        else:
            return equalalphanumeric(bch, fieldname, value)
    except KeyError as e:
        return equalalphanumeric(bch, fieldname, value)