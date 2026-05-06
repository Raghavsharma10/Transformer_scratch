def chapman_profile(Z0: float, zKM: np.ndarray, H: float):
    """
    Z0: altitude [km] of intensity peak
    zKM: altitude grid [km]
    H: scale height [km]

    example:
    pz = chapman_profile(110,np.arange(90,200,1),20)
    """
    return np.exp(.5*(1-(zKM-Z0)/H - np.exp((Z0-zKM)/H)))