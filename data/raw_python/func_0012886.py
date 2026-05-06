def mixerfields(data, commdct):
    """get mixer fields to diagram it"""
    objkey = "Connector:Mixer".upper()
    fieldlists = splittermixerfieldlists(data, commdct, objkey)
    return extractfields(data, commdct, objkey, fieldlists)