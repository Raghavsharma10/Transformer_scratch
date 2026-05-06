def setupz(Np: int, zmin: float, gridmin: float, gridmax: float) -> np.ndarray:
    """
    np: number of grid points
    zmin: minimum STEP SIZE at minimum grid altitude [km]
    gridmin: minimum altitude of grid [km]
    gridmax: maximum altitude of grid [km]
    """

    dz = _ztanh(Np, gridmin, gridmax)

    return np.insert(np.cumsum(dz)+zmin, 0, zmin)[:-1]