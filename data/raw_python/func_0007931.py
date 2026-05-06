def term(sign, lon):
    """ Returns the term for a sign and longitude. """
    terms = TERMS[sign]
    for (ID, a, b) in terms:
        if (a <= lon < b):
            return ID
    return None