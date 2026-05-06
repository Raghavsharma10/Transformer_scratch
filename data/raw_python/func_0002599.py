def Iaax(mt, x, *args):
    """ (Iä)x : Returns the present value of annuity-certain at the beginning of the first year 
    and increasing linerly. Arithmetically increasing annuity-anticipatory 
    """
    return Sx(mt, x) / Dx(mt, x)