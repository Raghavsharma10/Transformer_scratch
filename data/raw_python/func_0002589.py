def Axn(mt, x, n):
    """ (A^1)x:n : Returns the EPV (net single premium) of a term insurance. """
    return (mt.Mx[x] - mt.Mx[x + n]) / mt.Dx[x]