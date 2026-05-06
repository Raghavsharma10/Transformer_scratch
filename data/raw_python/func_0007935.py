def score(ID, sign, lon):
    """ Returns the score of an object on
    a sign and longitude.

    """
    info = getInfo(sign, lon)
    dignities = [dign for (dign, objID) in info.items() if objID == ID]
    return sum([SCORES[dign] for dign in dignities])