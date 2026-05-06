def get_fn_from_coords(coords, name=None):
    """ Given a set of coordinates, returns the standard filename.

    Parameters
    -----------
    coords : list
        [LLC.lat, LLC.lon, URC.lat, URC.lon]
    name : str (optional)
        An optional suffix to the filename.

    Returns
    -------
    fn : str
        The standard <filename>_<name>.tif with suffix (if supplied)
    """
    NS1 = ["S", "N"][coords[0] > 0]
    EW1 = ["W", "E"][coords[1] > 0]
    NS2 = ["S", "N"][coords[2] > 0]
    EW2 = ["W", "E"][coords[3] > 0]
    new_name = "%s%0.3g%s%0.3g_%s%0.3g%s%0.3g" % \
        (NS1, coords[0], EW1, coords[1], NS2, coords[2], EW2, coords[3])
    if name is not None:
        new_name += '_' + name
    return new_name.replace('.', 'o') + '.tif'