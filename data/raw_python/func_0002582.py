def tqx(mt, x, t):
    """ nqx : Returns the probability to die within n years at age x """
    return (mt.lx[x] - mt.lx[x + t]) / mt.lx[x]