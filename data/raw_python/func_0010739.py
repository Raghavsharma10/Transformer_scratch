def fit_bristow_campbell_params(tmin, tmax, pot_rad_daily, obs_rad_daily):
    """
    Fit the A and C parameters for the Bristow & Campbell (1984) model using observed daily
    minimum and maximum temperature and mean daily (e.g. aggregated from hourly values) solar
    radiation.

    Parameters
    ----------
    tmin : Series
        Observed daily minimum temperature.

    tmax : Series
        Observed daily maximum temperature.

    pot_rad_daily : Series
        Mean potential daily solar radiation.

    obs_rad_daily : Series
        Mean observed daily solar radiation.
    """
    def bc_absbias(ac):
        return np.abs(np.mean(bristow_campbell(df.tmin, df.tmax, df.pot, ac[0], ac[1]) - df.obs))

    df = pd.DataFrame(data=dict(tmin=tmin, tmax=tmax, pot=pot_rad_daily, obs=obs_rad_daily)).dropna(how='any')
    res = scipy.optimize.minimize(bc_absbias, [0.75, 2.4])  # i.e. we minimize the absolute bias

    return res.x