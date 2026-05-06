def get_lightcurve_from_file(file, *args, use_cols=None, skiprows=0,
                             verbosity=None,
                             **kwargs):
    """get_lightcurve_from_file(file, *args, use_cols=None, skiprows=0, **kwargs)

    Fits a light curve to the data contained in *file* using
    :func:`get_lightcurve`.

    **Parameters**

    file : str or file
        File or filename to load data from.
    use_cols : iterable or None, optional
        Iterable of columns to read from data file, or None to read all columns
        (default None).
    skiprows : number, optional
        Number of rows to skip at beginning of *file* (default 0)

    **Returns**

    out : dict
        See :func:`get_lightcurve`.
    """
    data = numpy.loadtxt(file, skiprows=skiprows, usecols=use_cols)
    if len(data) != 0:
        masked_data = numpy.ma.array(data=data, mask=None, dtype=float)
        return get_lightcurve(masked_data, *args,
                              verbosity=verbosity, **kwargs)
    else:
        verbose_print("{}: file contains no data points".format(file),
                      operation="coverage", verbosity=verbosity)
        return