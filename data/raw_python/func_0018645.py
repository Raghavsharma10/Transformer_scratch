def join_timeseries(base, overwrite, join_linear=None):
    """Join two sets of timeseries

    Parameters
    ----------
    base : :obj:`MAGICCData`, :obj:`pd.DataFrame`, filepath
        Base timeseries to use. If a filepath, the data will first be loaded from disk.

    overwrite : :obj:`MAGICCData`, :obj:`pd.DataFrame`, filepath
        Timeseries to join onto base. Any points which are in both `base` and
        `overwrite` will be taken from `overwrite`. If a filepath, the data will first
        be loaded from disk.

    join_linear : tuple of len(2)
        A list/array which specifies the period over which the two timeseries should
        be joined. The first element is the start time of the join period, the second
        element is the end time of the join period. In the join period (excluding the
        start and end times), output data will be a linear interpolation between (the
        annually interpolated) `base` and `overwrite` data. If None, no linear join
        will be done and any points in (the annually interpolated) `overwrite` data
        will simply overwrite any points in `base`.

    Returns
    -------
    :obj:`MAGICCData`
        The joint timeseries. The resulting data is linearly interpolated onto annual steps
    """
    if join_linear is not None:
        if len(join_linear) != 2:
            raise ValueError("join_linear must have a length of 2")

    if isinstance(base, str):
        base = MAGICCData(base)
    elif isinstance(base, MAGICCData):
        base = deepcopy(base)

    if isinstance(overwrite, str):
        overwrite = MAGICCData(overwrite)
    elif isinstance(overwrite, MAGICCData):
        overwrite = deepcopy(overwrite)

    result = _join_timeseries_mdata(base, overwrite, join_linear)

    return MAGICCData(result)