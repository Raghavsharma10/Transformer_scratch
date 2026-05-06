def rungtd1d(time: datetime,
             altkm: np.ndarray,
             glat: float, glon: float, *,
             f107a: float = None, f107: float = None, Ap: int = None) -> xarray.Dataset:
    """
    This is the "atomic" function looped by other functions
    """
    time = todatetime(time)
    # %% get solar parameters for date
    if f107a and f107a and Ap:
        pass
    else:
        f107Ap = gi.getApF107(time, smoothdays=81)
        f107a = f107Ap['f107s'].item()
        f107 = f107Ap['f107'].item()
        Ap = f107Ap['Ap'].item()
# %% dimensions
    altkm = np.atleast_1d(altkm)
    assert altkm.ndim == 1
    assert isinstance(glon, (int, float))
    assert isinstance(glat, (int, float))

# %%
    iyd = time.strftime('%y%j')
    altkm = np.atleast_1d(altkm)
# %%
    dens = np.empty((altkm.size, len(species)))
    temp = np.empty((altkm.size, len(ttypes)))
    for i, a in enumerate(altkm):
        cmd = [str(EXE),
               iyd, str(time.hour), str(time.minute), str(time.second),
               str(glat), str(glon),
               str(f107a), str(f107), str(Ap), str(a)]

        ret = subprocess.check_output(cmd,
                                      universal_newlines=True,
                                      stderr=subprocess.DEVNULL)
        f = io.StringIO(ret)
        dens[i, :] = np.genfromtxt(f, max_rows=1)
        temp[i, :] = np.genfromtxt(f, max_rows=1)

    dsf = {k: (('time', 'alt_km', 'lat', 'lon'), v[None, :, None, None]) for (k, v) in zip(species, dens.T)}
    dsf.update({'Tn':  (('time', 'alt_km', 'lat', 'lon'), temp[:, 1][None, :, None, None]),
                'Texo': (('time', 'alt_km', 'lat', 'lon'), temp[:, 0][None, :, None, None])})

    atmos = xarray.Dataset(dsf,
                           coords={'time': [time], 'alt_km': altkm, 'lat': [glat], 'lon': [glon], },
                           attrs={'Ap': Ap, 'f107': f107, 'f107a': f107a,
                                  'species': species})

    return atmos