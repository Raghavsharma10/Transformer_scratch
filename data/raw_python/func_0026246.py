def loopalt_gtd(time: datetime,
                glat: Union[float, np.ndarray], glon: Union[float, np.ndarray],
                altkm: Union[float, List[float], np.ndarray], *,
                f107a: float = None, f107: float = None, Ap: int = None) -> xarray.Dataset:
    """
    loop over location and time

    time: datetime or numpy.datetime64 or list of datetime or np.ndarray of datetime
    glat: float or 2-D np.ndarray
    glon: float or 2-D np.ndarray
    altkm: float or list or 1-D np.ndarray
    """
    glat = np.atleast_2d(glat)
    glon = np.atleast_2d(glon)
    assert glat.ndim == glon.ndim == 2

    times = np.atleast_1d(time)
    assert times.ndim == 1

    atmos = xarray.Dataset()

    for k, t in enumerate(times):
        print('computing', t)
        for i in range(glat.shape[0]):
            for j in range(glat.shape[1]):
                # atmos = xarray.concat((atmos, rungtd1d(t, altkm, glat[i,j], glon[i,j])),
                #                      data_vars='minimal',coords='minimal',dim='lon')
                atm = rungtd1d(t, altkm, glat[i, j], glon[i, j],
                               f107a=f107a, f107=f107, Ap=Ap)
                atmos = xarray.merge((atmos, atm))

    atmos.attrs = atm.attrs

    return atmos