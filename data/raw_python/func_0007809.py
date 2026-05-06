def getFixedStarList(IDs, date):
    """ Returns a list of fixed stars. """
    starList = [getFixedStar(ID, date) for ID in IDs]
    return FixedStarList(starList)