def tqxn(mt, x, n, t):
    """ n/qx : Probability to die in n years being alive at age x.
    Probability that x survives n year, and then dies in th subsequent t years """
    return tpx(mt, x, t) * qx(mt, x + n)