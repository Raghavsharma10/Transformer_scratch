def load(fnames, tag=None, sat_id=None, **kwargs):
    """Loads data using pysat.utils.load_netcdf4 .

    This routine is called as needed by pysat. It is not intended
    for direct user interaction.
    
    Parameters
    ----------
    fnames : array-like
        iterable of filename strings, full path, to data files to be loaded.
        This input is nominally provided by pysat itself.
    tag : string
        tag name used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.
    sat_id : string
        Satellite ID used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.
    **kwargs : extra keywords
        Passthrough for additional keyword arguments specified when 
        instantiating an Instrument object. These additional keywords
        are passed through to this routine by pysat.
    
    Returns
    -------
    data, metadata
        Data and Metadata are formatted for pysat. Data is a pandas 
        DataFrame while metadata is a pysat.Meta instance.
        
    Note
    ----
    Any additional keyword arguments passed to pysat.Instrument
    upon instantiation are passed along to this routine and through
    to the load_netcdf4 call.
    
    Examples
    --------
    ::
    
        inst = pysat.Instrument('sport', 'ivm')
        inst.load(2019,1)
    
        # create quick Instrument object for a new, random netCDF4 file
        # define filename template string to identify files
        # this is normally done by instrument code, but in this case
        # there is no built in pysat instrument support
        # presumes files are named default_2019-01-01.NC
        format_str = 'default_{year:04d}-{month:02d}-{day:02d}.NC'
        inst = pysat.Instrument('netcdf', 'pandas', 
                                custom_kwarg='test'
                                data_path='./',
                                format_str=format_str)
        inst.load(2019,1)
    
    """
    
    return pysat.utils.load_netcdf4(fnames, **kwargs)