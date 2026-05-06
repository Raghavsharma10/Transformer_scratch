def angSepVincenty(ra1, dec1, ra2, dec2):
    """
    Vincenty formula for distances on a sphere
    """
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    sin_dec1, cos_dec1 = np.sin(dec1_rad), np.cos(dec1_rad)
    sin_dec2, cos_dec2 = np.sin(dec2_rad), np.cos(dec2_rad)
    delta_ra = ra2_rad - ra1_rad
    cos_delta_ra, sin_delta_ra = np.cos(delta_ra), np.sin(delta_ra)

    diffpos = np.arctan2(np.sqrt((cos_dec2 * sin_delta_ra) ** 2 +
                         (cos_dec1 * sin_dec2 -
                          sin_dec1 * cos_dec2 * cos_delta_ra) ** 2),
                          sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * cos_delta_ra)

    return np.degrees(diffpos)