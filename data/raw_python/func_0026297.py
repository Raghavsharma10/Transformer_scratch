def open_bpchdataset(filename, fields=[], categories=[],
                     tracerinfo_file='tracerinfo.dat',
                     diaginfo_file='diaginfo.dat',
                     endian=">", decode_cf=True,
                     memmap=True, dask=True, return_store=False):
    """ Open a GEOS-Chem BPCH file output as an xarray Dataset.

    Parameters
    ----------
    filename : string
        Path to the output file to read in.
    {tracerinfo,diaginfo}_file : string, optional
        Path to the metadata "info" .dat files which are used to decipher
        the metadata corresponding to each variable in the output dataset.
        If not provided, will look for them in the current directory or
        fall back on a generic set.
    fields : list, optional
        List of a subset of variable names to return. This can substantially
        improve read performance. Note that the field here is just the tracer
        name - not the category, e.g. 'O3' instead of 'IJ-AVG-$_O3'.
    categories : list, optional
        List a subset of variable categories to look through. This can
        substantially improve read performance.
    endian : {'=', '>', '<'}, optional
        Endianness of file on disk. By default, "big endian" (">") is assumed.
    decode_cf : bool
        Enforce CF conventions for variable names, units, and other metadata
    default_dtype : numpy.dtype, optional
        Default datatype for variables encoded in file on disk (single-precision
        float by default).
    memmap : bool
        Flag indicating that data should be memory-mapped from disk instead of
        eagerly loaded into memory
    dask : bool
        Flag indicating that data reading should be deferred (delayed) to
        construct a task-graph for later execution
    return_store : bool
        Also return the underlying DataStore to the user

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the requested fields (or the entire file), with data
        contained in proxy containers for access later.
    store : xarray.AbstractDataStore
        Underlying DataStore which handles the loading and processing of
        bpch files on disk

    """

    store = BPCHDataStore(
        filename, fields=fields, categories=categories,
        tracerinfo_file=tracerinfo_file,
        diaginfo_file=diaginfo_file, endian=endian,
        use_mmap=memmap, dask_delayed=dask
    )
    ds = xr.Dataset.load_store(store)
    # Record what the file object underlying the store which we culled this
    # Dataset from is so that we can clean it up later
    ds._file_obj = store._bpch

    # Handle CF corrections
    if decode_cf:
        decoded_vars = OrderedDict()
        rename_dict = {}
        for v in ds.variables:
            cf_name = cf.get_valid_varname(v)
            rename_dict[v] = cf_name
            new_var = cf.enforce_cf_variable(ds[v])
            decoded_vars[cf_name] = new_var
        ds = xr.Dataset(decoded_vars, attrs=ds.attrs.copy())

        # ds.rename(rename_dict, inplace=True)

        # TODO: There's a bug with xr.decode_cf which eagerly loads data.
        #       Re-enable this once that bug is fixed
        # Note that we do not need to decode the times because we explicitly
        # kept track of them as we parsed the data.
        # ds = xr.decode_cf(ds, decode_times=False)

    # Set attributes for CF conventions
    ts = get_timestamp()
    ds.attrs.update(dict(
        Conventions='CF1.6',
        source=filename,
        tracerinfo=tracerinfo_file,
        diaginfo=diaginfo_file,
        filetype=store._bpch.filetype,
        filetitle=store._bpch.filetitle,
        history=(
            "{}: Processed/loaded by xbpch-{} from {}"
            .format(ts, ver, filename)
        ),
    ))

    # To immediately load the data from the BPCHDataProxy paylods, need
    # to execute ds.data_vars for some reason...
    if return_store:
        return ds, store
    else:
        return ds