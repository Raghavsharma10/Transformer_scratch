def Iaxn(mt, x, n, *args):
    """ during a term certain, IAn """
    return (Sx(mt, x + 1) - Sx(mt, x + n + 1) - n * Nx(mt, x + n + 1)) / Dx(mt, x)