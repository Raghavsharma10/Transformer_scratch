def find_near_pol_ang(aryEmpPlrAng, aryExpPlrAng):
    """Return index of nearest expected polar angle.

    Parameters
    ----------
    aryEmpPlrAng : 1D numpy array
        Empirically found polar angle estimates
    aryExpPlrAng : 1D numpy array
        Theoretically expected polar angle estimates

    Returns
    -------
    aryXCrds : 1D numpy array
        Indices of nearest theoretically expected polar angle.
    aryYrds : 1D numpy array
        Distances to nearest theoretically expected polar angle.
    """

    dist = np.abs(np.subtract(aryEmpPlrAng[:, None],
                              aryExpPlrAng[None, :]))

    return np.argmin(dist, axis=-1), np.min(dist, axis=-1)