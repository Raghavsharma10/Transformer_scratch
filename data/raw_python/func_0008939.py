def get_fn(elev, name=None):
    """
    Determines the standard filename for a given GeoTIFF Layer.

    Parameters
    -----------
    elev : GdalReader.raster_layer
        A raster layer from the GdalReader object.
    name : str (optional)
        An optional suffix to the filename.
    Returns
    -------
    fn : str
        The standard <filename>_<name>.tif with suffix (if supplied)
    """
    gcs = elev.grid_coordinates
    coords = [gcs.LLC.lat, gcs.LLC.lon, gcs.URC.lat, gcs.URC.lon]
    return get_fn_from_coords(coords, name)