def qx(mt, x):
    """ qx: Returns the probability that a life aged x dies before 1 year
            With the convention: the true probability is qx/1000
    Args:
        mt: the mortality table
        x: the age as integer number.
    """
    if x < len(mt.qx):
        return mt.qx[x]
    else:
        return 0