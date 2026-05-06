def getsinIm(alat):
    """Computes sinIm from modified apex latitude.

    Parameters
    ==========
    alat : array_like
        Modified apex latitude

    Returns
    =======
    sinIm : ndarray or float

    """

    alat = np.float64(alat)

    return 2*np.sin(np.radians(alat))/np.sqrt(4 - 3*np.cos(np.radians(alat))**2)