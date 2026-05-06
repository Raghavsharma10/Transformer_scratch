def dewpoint_temperature(temp, hum):
    """computes the dewpoint temperature

    Parameters
    ----
    temp :      temperature [K]
    hum :       relative humidity
    

    Returns
        dewpoint temperature in K
    """
    assert(temp.shape == hum.shape)

    vap_press = vapor_pressure(temp, hum)

    positives = np.array(temp >= 273.15)
    dewpoint_temp = temp.copy() * np.nan
    dewpoint_temp[positives] = 243.12 * np.log(vap_press[positives] / 6.112) / (17.62 - np.log(vap_press[positives] / 6.112))
    dewpoint_temp[~positives] = 272.62 * np.log(vap_press[~positives] / 6.112) / (22.46 - np.log(vap_press[~positives] / 6.112))

    return dewpoint_temp + 273.15