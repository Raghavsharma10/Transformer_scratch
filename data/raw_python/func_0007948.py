def termLons(TERMS):
    """ Returns a list with the absolute longitude 
    of all terms.
    
    """
    res = []
    for i, sign in enumerate(SIGN_LIST):
        termList = TERMS[sign]
        res.extend([
            ID,
            sign,
            start + 30 * i,
        ] for (ID, start, end) in termList)
    return res