def run(time: datetime, altkm: float,
        glat: Union[float, np.ndarray], glon: Union[float, np.ndarray], *,
        f107a: float = None, f107: float = None, Ap: int = None) -> xarray.Dataset:
    """
    loops the rungtd1d function below. Figure it's easier to troubleshoot in Python than Fortran.
    """
    glat = np.atleast_2d(glat)
    glon = np.atleast_2d(glon)  # has to be here
# %% altitude 1-D
    if glat.size == 1 and glon.size == 1 and isinstance(time, (str, date, datetime, np.datetime64)):
        atmos = rungtd1d(time, altkm, glat.squeeze()[()], glon.squeeze()[()],
                         f107a=f107a, f107=f107, Ap=Ap)
# %% lat/lon grid at 1 altitude
    else:
        atmos = loopalt_gtd(time, glat, glon, altkm,
                            f107a=f107a, f107=f107, Ap=Ap)

    return atmos