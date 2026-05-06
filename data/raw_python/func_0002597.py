def Iaaxn(mt, x, n, *args):
    """ during a term certain, IAn """
    return (Sx(mt, x) - Sx(nt, x + n) - n * Nx(nt, x + n)) / Dx(nt, x)