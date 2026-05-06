def plantloopfields(data, commdct):
    """get plantloop fields to diagram it"""
    fieldlists = plantloopfieldlists(data)
    objkey = 'plantloop'.upper()
    return extractfields(data, commdct, objkey, fieldlists)