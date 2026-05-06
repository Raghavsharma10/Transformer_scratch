def get_sun_times(dates, lon, lat, time_zone):
    """Computes the times of sunrise, solar noon, and sunset for each day.

    Parameters
    ----
    dates:      datetime
    lat :       latitude in DecDeg
    lon :       longitude in DecDeg
    time_zone : timezone
    

    Returns
    ----
    DataFrame:  [sunrise, sunnoon, sunset, day length] in dec hours
    """

    df = pd.DataFrame(index=dates, columns=['sunrise', 'sunnoon', 'sunset', 'daylength'])

    doy = np.array([(d - d.replace(day=1, month=1)).days + 1 for d in df.index])  # day of year

    # Day angle and declination after Bourges (1985):
    day_angle_b = np.deg2rad((360. / 365.25) * (doy - 79.346))
    
    declination = np.deg2rad(
        0.3723 + 23.2567 * np.sin(day_angle_b) - 0.7580 * np.cos(day_angle_b)
        + 0.1149 * np.sin(2*day_angle_b) + 0.3656 * np.cos(2*day_angle_b)
        - 0.1712 * np.sin(3*day_angle_b) + 0.0201 * np.cos(3*day_angle_b)
    )
    
    # Equation of time with day angle after Spencer (1971):
    day_angle_s = 2 * np.pi * (doy - 1) / 365.
    eq_time = 12. / np.pi * (
        0.000075 +
        0.001868 * np.cos(  day_angle_s) - 0.032077 * np.sin(  day_angle_s) -
        0.014615 * np.cos(2*day_angle_s) - 0.040849 * np.sin(2*day_angle_s)
        )
    
    #
    standard_meridian = time_zone * 15.
    delta_lat_time = (lon - standard_meridian) * 24. / 360.
    
    omega_nul_arg = -np.tan(np.deg2rad(lat)) * np.tan(declination)
    omega_nul = np.arccos(omega_nul_arg)
    sunrise = 12. * (1. - (omega_nul) / np.pi) - delta_lat_time - eq_time
    sunset  = 12. * (1. + (omega_nul) / np.pi) - delta_lat_time - eq_time

    # as an approximation, solar noon is independent of the below mentioned
    # cases:
    sunnoon  = 12. * (1.) - delta_lat_time - eq_time
    
    # $kf 2015-11-13: special case midnight sun and polar night
    # CASE 1: MIDNIGHT SUN
    # set sunrise and sunset to values that would yield the maximum day
    # length even though this a crude assumption
    pos = omega_nul_arg < -1
    sunrise[pos] = sunnoon[pos] - 12
    sunset[pos]  = sunnoon[pos] + 12

    # CASE 2: POLAR NIGHT
    # set sunrise and sunset to values that would yield the minmum day
    # length even though this a crude assumption
    pos = omega_nul_arg > 1
    sunrise[pos] = sunnoon[pos]
    sunset[pos]  = sunnoon[pos]

    daylength = sunset - sunrise
        
    # adjust if required
    sunrise[sunrise < 0] += 24
    sunset[sunset > 24] -= 24

    df.sunrise = sunrise
    df.sunnoon = sunnoon
    df.sunset = sunset
    df.daylength = daylength

    return df