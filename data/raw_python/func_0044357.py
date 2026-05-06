def read_ascii_catalog(filename, format_, unit=None):
    """
    Read an ASCII catalog file using Astropy.

    This routine is used by pymoctool to load coordinates from a
    catalog file in order to generate a MOC representation.
    """

    catalog = ascii.read(filename, format=format_)
    columns = catalog.columns

    if 'RA' in columns and 'Dec' in columns:
        if unit is None:
            unit = (hour, degree)

        coords = SkyCoord(catalog['RA'],
                          catalog['Dec'],
                          unit=unit,
                          frame='icrs')

    elif 'Lat' in columns and 'Lon' in columns:
        if unit is None:
            unit = (degree, degree)

        coords = SkyCoord(catalog['Lon'],
                          catalog['Lat'],
                          unit=unit,
                          frame='galactic')

    else:
        raise Exception('columns RA,Dec or Lon,Lat not found')

    return coords