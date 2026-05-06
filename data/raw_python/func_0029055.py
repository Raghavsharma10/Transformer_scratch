def damping(temp, relhum, freq, pres=101325):
    """
    Calculates the damping factor for sound in dB/m
    depending on temperature, humidity and sound frequency.
    Source: http://www.sengpielaudio.com/LuftdaempfungFormel.htm

    temp: Temperature in degrees celsius
    relhum: Relative humidity as percentage, e.g. 50
    freq: Sound frequency in herz
    pres: Atmospheric pressure in kilopascal
    """
    temp += 273.15  # convert to kelvin
    pres = pres / 101325.0  # convert to relative pressure
    c_humid = 4.6151 - 6.8346 * pow((273.15 / temp), 1.261)
    hum = relhum * pow(10.0, c_humid) * pres
    tempr = temp / 293.15  # convert to relative air temp (re 20 deg C)
    frO = pres * (24.0 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum))
    frN = (pres * pow(tempr, -0.5) * (9.0 + 280.0 * hum * math.exp(-4.17 *
        (pow(tempr, (-1.0 / 3.0)) - 1.0))))
    damp = (8.686 * freq * freq * (
            1.84e-11 * (1.0 / pres) * math.sqrt(tempr) +
            pow(tempr, -2.5) *
            (
                0.01275 * (math.exp(-2239.1 / temp) * 1.0 /
                (frO + freq * freq / frO)) +
                0.1068 * (
                    math.exp(-3352 / temp) * 1.0 /
                    (frN + freq * freq / frN)
                )
            )
        )
    )
    return damp