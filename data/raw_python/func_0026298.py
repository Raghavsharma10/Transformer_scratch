def open_mfbpchdataset(paths, concat_dim='time', compat='no_conflicts',
                       preprocess=None, lock=None, **kwargs):
    """ Open multiple bpch files as a single dataset.

    You must have dask installed for this to work, as this greatly
    simplifies issues relating to multi-file I/O.

    Also, please note that this is not a very performant routine. I/O is still
    limited by the fact that we need to manually scan/read through each bpch
    file so that we can figure out what its contents are, since that metadata
    isn't saved anywhere. So this routine will actually sequentially load
    Datasets for each bpch file, then concatenate them along the "time" axis.
    You may wish to simply process each file individually, coerce to NetCDF,
    and then ingest through xarray as normal.

    Parameters
    ----------
    paths : list of strs
        Filenames to load; order doesn't matter as they will be
        lexicographically sorted before we read in the data
    concat_dim : str, default='time'
        Dimension to concatenate Datasets over. We default to "time" since this
        is how GEOS-Chem splits output files
    compat : str (optional)
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    preprocess : callable (optional)
        A pre-processing function to apply to each Dataset prior to
        concatenation
    lock : False, True, or threading.Lock (optional)
        Passed to :py:func:`dask.array.from_array`. By default, xarray
        employs a per-variable lock when reading data from NetCDF files,
        but this model has not yet been extended or implemented for bpch files
        and so this is not actually used. However, it is likely necessary
        before dask's multi-threaded backend can be used
    **kwargs : optional
        Additional arguments to pass to :py:func:`xbpch.open_bpchdataset`.
    
    """

    from xarray.backends.api import _MultiFileCloser

    # TODO: Include file locks?

    # Check for dask
    dask = kwargs.pop('dask', False)
    if not dask:
        raise ValueError("Reading multiple files without dask is not supported")
    kwargs['dask'] = True

    # Add th

    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError("No paths to files were passed into open_mfbpchdataset")

    datasets = [open_bpchdataset(filename, **kwargs)
                for filename in paths]
    bpch_objs = [ds._file_obj for ds in datasets]

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    # Concatenate over time
    combined = xr.auto_combine(datasets, compat=compat, concat_dim=concat_dim)

    combined._file_obj = _MultiFileCloser(bpch_objs)
    combined.attrs = datasets[0].attrs
    ts = get_timestamp()
    fns_str = " ".join(paths)
    combined.attrs['history'] = (
        "{}: Processed/loaded by xbpch-{} from {}"
        .format(ts, ver, fns_str)
    )


    return combined