def hitail(E: np.ndarray, diffnumflux: np.ndarray, isimE0: np.ndarray, E0: np.ndarray,
           Bhf: np.ndarray, bh: float, verbose: int = 0):
    """
    strickland 1993 said 0.2, but 0.145 gives better match to peak flux at 2500 = E0
    """
    Bh = np.empty_like(E0)
    for iE0 in np.arange(E0.size):
        Bh[iE0] = Bhf[iE0]*diffnumflux[isimE0[iE0], iE0]  # 4100.
    # bh = 4                   #2.9
    het = Bh*(E[:, None] / E0)**-bh
    het[E[:, None] < E0] = 0.
    if verbose > 0:
        print('Bh: ' + (' '.join('{:0.1f}'.format(b) for b in Bh)))
    return het