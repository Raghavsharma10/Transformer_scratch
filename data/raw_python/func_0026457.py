def getcosIm(alat):
    """Computes cosIm from modified apex latitude.

    Parameters
    ==========
    alat : array_like
        Modified apex latitude

    Returns
    =======
    cosIm : ndarray or float

    """

    alat = np.float64(alat)

    return np.cos(np.radians(alat))/np.sqrt(4 - 3*np.cos(np.radians(alat))**2)