def ang_sep(row, ra1, dec1):
    """
    Calculate angular separation between two coordinates
    Uses Vicenty Formula (http://en.wikipedia.org/wiki/Great-circle_distance) and adapts from astropy's SkyCoord
    Written to be used within the Database.search() method

    Parameters
    ----------
    row: dict, pandas Row
        Coordinate structure containing ra and dec keys in decimal degrees
    ra1: float
        RA to compare with, in decimal degrees
    dec1: float
        Dec to compare with, in decimal degrees

    Returns
    -------
        Angular distance, in degrees, between the coordinates
    """

    factor = math.pi / 180
    sdlon = math.sin((row['ra'] - ra1) * factor)  # RA is longitude
    cdlon = math.cos((row['ra'] - ra1) * factor)
    slat1 = math.sin(dec1 * factor)  # Dec is latitude
    slat2 = math.sin(row['dec'] * factor)
    clat1 = math.cos(dec1 * factor)
    clat2 = math.cos(row['dec'] * factor)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    numerator = math.sqrt(num1 ** 2 + num2 ** 2)
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(numerator, denominator) / factor