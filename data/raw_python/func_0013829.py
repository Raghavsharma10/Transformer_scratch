def load(fnames, tag=None, sat_id=None, 
         fake_daily_files_from_monthly=False,
         flatten_twod=True):
    """Load NASA CDAWeb CDF files.
    
    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Parameters
    ------------
    fnames : (pandas.Series)
        Series of filenames
    tag : (str or NoneType)
        tag or None (default=None)
    sat_id : (str or NoneType)
        satellite id or None (default=None)
    fake_daily_files_from_monthly : bool
        Some CDAWeb instrument data files are stored by month, interfering
        with pysat's functionality of loading by day. This flag, when true,
        parses of daily dates to monthly files that were added internally
        by the list_files routine, when flagged. These dates are
        used here to provide data by day. 

    Returns
    ---------
    data : (pandas.DataFrame)
        Object containing satellite data
    meta : (pysat.Meta)
        Object containing metadata such as column names and units
        
    Examples
    --------
    ::
    
        # within the new instrument module, at the top level define
        # a new variable named load, and set it equal to this load method
        # code below taken from cnofs_ivm.py.
        
        # support load routine
        # use the default CDAWeb method
        load = cdw.load

        
    """
    
    import pysatCDF

    if len(fnames) <= 0 :
        return pysat.DataFrame(None), None
    else:
        # going to use pysatCDF to load the CDF and format
        # data and metadata for pysat using some assumptions.
        # Depending upon your needs the resulting pandas DataFrame may
        # need modification 
        # currently only loads one file, which handles more situations via pysat
        # than you may initially think
        
        if fake_daily_files_from_monthly:
            # parse out date from filename
            fname = fnames[0][0:-11]
            date = pysat.datetime.strptime(fnames[0][-10:], '%Y-%m-%d')
            with pysatCDF.CDF(fname) as cdf:
                # convert data to pysat format
                data, meta = cdf.to_pysat(flatten_twod=flatten_twod)
                # select data from monthly
                data = data.ix[date:date+pds.DateOffset(days=1) - pds.DateOffset(microseconds=1),:]
                return data, meta     
        else:
            # basic data return 
            with pysatCDF.CDF(fnames[0]) as cdf:     
                return cdf.to_pysat(flatten_twod=flatten_twod)