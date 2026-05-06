def splitterfields(data, commdct):
    """get splitter fields to diagram it"""
    objkey = "Connector:Splitter".upper()
    fieldlists = splittermixerfieldlists(data, commdct, objkey)
    return extractfields(data, commdct, objkey, fieldlists)