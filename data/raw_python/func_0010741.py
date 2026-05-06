def fit_angstroem_params(ssd, day_length, pot_rad_daily, obs_rad_daily):
    """
    Fit the a and b parameters for the Angstroem (1924) model using observed daily
    sunshine duration and mean daily (e.g. aggregated from hourly values) solar
    radiation.

    Parameters
    ----------
    ssd : Series
        Observed daily sunshine duration.

    day_length : Series
        Day lengths as calculated by ``calc_sun_times``.

    pot_rad_daily : Series
        Mean potential daily solar radiation.

    obs_rad_daily : Series
        Mean observed daily solar radiation.
    """
    df = pd.DataFrame(data=dict(ssd=ssd, day_length=day_length, pot=pot_rad_daily, obs=obs_rad_daily)).dropna(how='any')

    def angstroem_opt(x, a, b):
        return angstroem(x[0], x[1], x[2], a, b)

    x = np.array([df.ssd, df.day_length, df.pot])
    popt, pcov = scipy.optimize.curve_fit(angstroem_opt, x, df.obs, p0=[0.25, 0.75])

    return popt