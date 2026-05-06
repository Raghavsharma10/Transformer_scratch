def quick_stewart(input_geojson_points, variable_name, span,
                  beta=2, typefct='exponential',nb_class=None,
                  nb_pts=10000, resolution=None, mask=None,
                  user_defined_breaks=None, variable_name2=None,
                  output="GeoJSON", **kwargs):
    """
    Function acting as a one-shot wrapper around SmoothStewart object.
    Read a file of point values and optionnaly a mask file,
    return the smoothed representation as GeoJSON or GeoDataFrame.

    Parameters
    ----------
    input_geojson_points : str
        Path to file to use as input (Points/Polygons) or GeoDataFrame object,
        must contains a relevant numerical field.
    variable_name : str
        The name of the variable to use (numerical field only).
    span : int
        The span (meters).
    beta : float
        The beta!
    typefct : str, optionnal
        The type of function in {"exponential", "pareto"} (default: "exponential").
    nb_class : int, optionnal
        The number of class, if unset will most likely be 8
        (default: None)
    nb_pts: int, optionnal
        The number of points to use for the underlying grid.
        (default: 10000)
    resolution : int, optionnal
        The resolution to use (in meters), if not set a default
        resolution will be used in order to make a grid containing around
        10000 pts (default: None).
    mask : str, optionnal
        Path to the file (Polygons only) to use as clipping mask,
        can also be a GeoDataFrame (default: None).
    user_defined_breaks : list or tuple, optionnal
        A list of ordered break to use to construct the contours
        (override `nb_class` value if any, default: None).
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

    >>> result = quick_stewart("some_file.geojson", "some_variable",
                               span=12500, beta=3, typefct="exponential")

    More options, returning a GeoDataFrame:

    >>> smooth_gdf = quick_stewart("some_file.geojson", "some_variable",
                                   span=12500, beta=3, typefct="pareto",
                                   output="GeoDataFrame")
    """ 
    return SmoothStewart(
            input_geojson_points,
            variable_name,
            span,
            beta,
            typefct,
            nb_pts,
            resolution,
            variable_name2,
            mask,
            **kwargs
            ).render(
                nb_class=nb_class,
                user_defined_breaks=user_defined_breaks,
                output=output)