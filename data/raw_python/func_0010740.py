def angstroem(ssd, day_length, pot_rad_daily, a, b):
    """
    Calculate mean daily radiation from observed sunshine duration according to Angstroem (1924).

    Parameters
    ----------
    ssd : Series
        Observed daily sunshine duration.

    day_length : Series
        Day lengths as calculated by ``calc_sun_times``.

    pot_rad_daily : Series
        Mean potential daily solar radiation.

    a : float
        First parameter for the Angstroem model (originally 0.25).

    b : float
        Second parameter for the Angstroem model (originally 0.75).
    """
    if isinstance(a, pd.Series):
        months = ssd.index.month
        a = a.loc[months].values
        b = b.loc[months].values

    glob_day = (a + b * ssd / day_length) * pot_rad_daily

    return glob_day