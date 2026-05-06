def Cx(mt, x):
    """ Return the Cx """   
    return ((1 / (1 + mt.i)) ** (x + 1)) * mt.dx[x] * ((1 + mt.i) ** 0.5)