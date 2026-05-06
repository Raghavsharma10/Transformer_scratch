def vapor_pressure(temp, hum):
    """
    Calculates vapor pressure from temperature and humidity after Sonntag (1990).

    Args:
        temp: temperature values
        hum: humidity value(s). Can be scalar (e.g. for calculating saturation vapor pressure).

    Returns:
        Vapor pressure in hPa.
    """

    if np.isscalar(hum):
        hum = np.zeros(temp.shape) + hum

    assert(temp.shape == hum.shape)

    positives = np.array(temp >= 273.15)
    vap_press = np.zeros(temp.shape) * np.nan
    vap_press[positives] = 6.112 * np.exp((17.62 * (temp[positives] - 273.15)) / (243.12 + (temp[positives] - 273.15))) * hum[positives] / 100.
    vap_press[~positives] = 6.112 * np.exp((22.46 * (temp[~positives] - 273.15)) / (272.62  + (temp[~positives] - 273.15))) * hum[~positives] / 100.

    return vap_press