def AExn(mt, x, n):
    """ AExn : Returns the EPV of a endowment insurance. 
    An endowment insurance provides a combination of a term insurance and a pure endowment 
    """
    return (mt.Mx[x] - mt.Mx[x + n]) / mt.Dx[x] + mt.Dx[x + n] / mt.Dx[x]