def potential_radiation(dates, lon, lat, timezone, terrain_slope=0, terrain_slope_azimuth=0,
                        cloud_fraction=0, split=False):
    """
    Calculate potential shortwave radiation for a specific location and time.

    This routine calculates global radiation as described in:
    Liston, G. E. and Elder, K. (2006): A Meteorological Distribution System for
    High-Resolution Terrestrial Modeling (MicroMet), J. Hydrometeorol., 7, 217–234.

    Corrections for eccentricity are carried out following:
    Paltridge, G.W., Platt, C.M.R., 1976. Radiative processes in Meteorology and Climatology.
    Elsevier Scientific Publishing Company, Amsterdam, Oxford, New York.

    Parameters
    ----------
    dates : DatetimeIndex or array-like
        The dates for which potential radiation shall be calculated
    lon : float
        Longitude (degrees)
    lat : float
        Latitude (degrees)
    timezone : float
        Time zone
    terrain_slope : float, default 0
        Terrain slope as defined in Liston & Elder (2006) (eq. 12)
    terrain_slope_azimuth : float, default 0
        Terrain slope azimuth as defined in Liston & Elder (2006) (eq. 13)
    cloud_fraction : float, default 0
        Cloud fraction between 0 and 1
    split : boolean, default False
        If True, return a DataFrame containing direct and diffuse radiation,
        otherwise return a Series containing total radiation
    """
    solar_constant = 1367.
    days_per_year = 365.25
    tropic_of_cancer = np.deg2rad(23.43697)
    solstice = 173.0

    dates = pd.DatetimeIndex(dates)
    dates_hour = np.array(dates.hour)
    dates_minute = np.array(dates.minute)
    day_of_year = np.array(dates.dayofyear)

    # compute solar decline in rad
    solar_decline = tropic_of_cancer * np.cos(2.0 * np.pi * (day_of_year - solstice) / days_per_year)

    # compute the sun hour angle in rad
    standard_meridian = timezone * 15.
    delta_lat_time = (lon - standard_meridian) * 24. / 360.
    hour_angle = np.pi * (((dates_hour + dates_minute / 60. + delta_lat_time) / 12.) - 1.)

    # get solar zenith angle
    cos_solar_zenith = (np.sin(solar_decline) * np.sin(np.deg2rad(lat))
                        + np.cos(solar_decline) * np.cos(np.deg2rad(lat)) * np.cos(hour_angle))
    cos_solar_zenith = cos_solar_zenith.clip(min=0)
    solar_zenith_angle = np.arccos(cos_solar_zenith)

    # compute transmissivities for direct and diffus radiation using cloud fraction
    transmissivity_direct = (0.6 + 0.2 * cos_solar_zenith) * (1.0 - cloud_fraction)
    transmissivity_diffuse = (0.3 + 0.1 * cos_solar_zenith) * cloud_fraction

    # modify solar constant for eccentricity
    beta = 2. * np.pi * (day_of_year / days_per_year)
    radius_ratio = (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta)
                    + 0.000719 * np.cos(2. * beta) + 0.000077 * np.sin(2 * beta))
    solar_constant_times_radius_ratio = solar_constant * radius_ratio

    mu = np.arcsin(np.cos(solar_decline) * np.sin(hour_angle) / np.sin(solar_zenith_angle))
    cosi = (np.cos(terrain_slope) * cos_solar_zenith
            + np.sin(terrain_slope) * np.sin(solar_zenith_angle) * np.cos(mu - terrain_slope_azimuth))

    # get total shortwave radiation
    direct_radiation = solar_constant_times_radius_ratio * transmissivity_direct * cosi
    diffuse_radiation = solar_constant_times_radius_ratio * transmissivity_diffuse * cos_solar_zenith
    direct_radiation = direct_radiation.clip(min=0)

    df = pd.DataFrame(index=dates, data=dict(direct=direct_radiation, diffuse=diffuse_radiation))

    if split:
        return df
    else:
        return df.direct + df.diffuse