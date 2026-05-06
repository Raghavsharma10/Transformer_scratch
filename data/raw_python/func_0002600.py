def Iax(mt, x, *args):
    """ (Ia)x : Returns the present value of annuity-certain at the end of the first year 
    and increasing linerly. Arithmetically increasing annuity-late 
    """
    return Sx(mt, x + 1) / Dx(mt, x)