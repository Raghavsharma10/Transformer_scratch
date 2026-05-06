def isPeregrine(ID, sign, lon):
    """ Returns if an object is peregrine
    on a sign and longitude.

    """
    info = getInfo(sign, lon)
    for dign, objID in info.items():
        if dign not in ['exile', 'fall'] and ID == objID:
            return False
    return True