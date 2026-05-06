def tpx(mt, x, t):
    """ tpx : Returns the probability that x will survive within t years """
    """ npx : Returns n years survival probability at age x """
    return mt.lx[x + t] / mt.lx[x]