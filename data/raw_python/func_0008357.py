def _equation_of_time(t):
    """
    Find the difference between apparent and mean solar time

    Parameters
    ----------
    t : `~astropy.time.Time`
        times (array)

    Returns
    ----------
    ret1 : `~astropy.units.Quantity`
        the equation of time
    """

    # Julian centuries since J2000.0
    T = (t - Time("J2000")).to(u.year).value / 100

    # obliquity of ecliptic (Meeus 1998, eq 22.2)
    poly_pars = (84381.448, 46.8150, 0.00059, 0.001813)
    eps = u.Quantity(polyval(T, poly_pars), u.arcsec)
    y = np.tan(eps/2)**2

    # Sun's mean longitude (Meeus 1998, eq 25.2)
    poly_pars = (280.46646, 36000.76983, 0.0003032)
    L0 = u.Quantity(polyval(T, poly_pars), u.deg)

    # Sun's mean anomaly (Meeus 1998, eq 25.3)
    poly_pars = (357.52911, 35999.05029, 0.0001537)
    M = u.Quantity(polyval(T, poly_pars), u.deg)

    # eccentricity of Earth's orbit (Meeus 1998, eq 25.4)
    poly_pars = (0.016708634, -0.000042037, -0.0000001267)
    e = polyval(T, poly_pars)

    # equation of time, radians (Meeus 1998, eq 28.3)
    eot = (y * np.sin(2*L0) - 2*e*np.sin(M) + 4*e*y*np.sin(M)*np.cos(2*L0) -
           0.5*y**2 * np.sin(4*L0) - 5*e**2 * np.sin(2*M)/4) * u.rad
    return eot.to(u.hourangle)