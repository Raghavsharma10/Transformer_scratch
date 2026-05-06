def getInfo(sign, lon):
    """ Returns the complete essential dignities
    for a sign and longitude.

    """
    return {
        'ruler': ruler(sign),
        'exalt': exalt(sign),
        'dayTrip': dayTrip(sign),
        'nightTrip': nightTrip(sign),
        'partTrip': partTrip(sign),
        'term': term(sign, lon),
        'face': face(sign, lon),
        'exile': exile(sign),
        'fall': fall(sign)
    }