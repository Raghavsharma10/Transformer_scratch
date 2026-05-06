def quick_idw(input_geojson_points, variable_name, power, nb_class,
              nb_pts=10000, resolution=None, disc_func=None,
              mask=None, user_defined_breaks=None,
              variable_name2=None, output='GeoJSON', **kwargs):
    """
    Function acting as a one-shot wrapper around SmoothIdw object.
    Read a file of point values and optionnaly a mask file,
    return the smoothed representation as GeoJSON or GeoDataFrame.

    Parameters
    ----------
    input_geojson_points : str
        Path to file to use as input (Points/Polygons) or GeoDataFrame object,
        must contains a relevant numerical field.
    variable_name : str
        The name of the variable to use (numerical field only).
    power : int or float
        The power of the function.
    nb_class : int, optionnal
        The number of class, if unset will most likely be 8.
        (default: None)
    nb_pts: int, optionnal
        The number of points to use for the underlying grid.
        (default: 10000)
    resolution : int, optionnal
        The resolution to use (in meters), if not set a default
        resolution will be used in order to make a grid containing around
        10000 pts (default: None).
    disc_func: str, optionnal
        The name of the classification function to be used to decide on which
        break values to use to create the contour layer.
        (default: None)
    mask : str, optionnal
        Path to the file (Polygons only) to use as clipping mask,
        can also be a GeoDataFrame (default: None).
    user_defined_breaks : list or tuple, optionnal
        A list of ordered break to use to construct the contours
        (overrides `nb_class` and `disc_func` values if any, default: None).
    variable_name2 : str, optionnal
        The name of the 2nd variable to use (numerical field only); values
        computed from this variable will be will be used as to divide
        values computed from the first variable (default: None)
    output : string, optionnal
        The type of output expected (not case-sensitive)
        in {"GeoJSON", "GeoDataFrame"} (default: "GeoJSON").

    Returns
    -------
    smoothed_result : bytes or GeoDataFrame,
        The result, dumped as GeoJSON (utf-8 encoded) or as a GeoDataFrame.


    Examples
    --------
    Basic usage, output to raw geojson (bytes):

    >>> result = quick_idw("some_file.geojson", "some_variable", power=2)

    More options, returning a GeoDataFrame:

    >>> smooth_gdf = quick_stewart("some_file.geojson", "some_variable",
                                   nb_class=8, disc_func="percentiles",
                                   output="GeoDataFrame")
    """

    return SmoothIdw(input_geojson_points,
                    variable_name,
                    power,
                    nb_pts,
                    resolution,
                    variable_name2,
                    mask,
                    **kwargs
                    ).render(nb_class=nb_class,
                      disc_func=disc_func,
                      user_defined_breaks=user_defined_breaks,
                      output=output)