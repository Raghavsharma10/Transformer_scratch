def parse_fn(fn):
    """ This parses the file name and returns the coordinates of the tile

    Parameters
    -----------
    fn : str
        Filename of a GEOTIFF

    Returns
    --------
    coords = [LLC.lat, LLC.lon, URC.lat, URC.lon]
    """
    try:
        parts = os.path.splitext(os.path.split(fn)[-1])[0].replace('o', '.')\
            .split('_')[:2]
        coords = [float(crds)
                  for crds in re.split('[NSEW]', parts[0] + parts[1])[1:]]
    except:
        coords = [np.nan] * 4
    return coords