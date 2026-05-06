def tAx(mt, x, t):
    """ n/Ax : Returns the EPV (net single premium) of a deferred whole life insurance. """
    return mt.Mx[x + t] / mt.Dx[x]