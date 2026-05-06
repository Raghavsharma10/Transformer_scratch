def list_files(tag=None, sat_id=None, data_path=None, format_str=None):
    """Return a Pandas Series of every file for chosen satellite data

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are '1min' and '5min'.
        (default=None)
    sat_id : (string or NoneType)
        Specifies the satellite ID for a constellation.  Not used.
        (default=None)
    data_path : (string or NoneType)
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)
    format_str : (string or NoneType)
        User specified file format.  If None is specified, the default
        formats associated with the supplied tags are used. (default=None)

    Returns
    --------
    pysat.Files.from_os : (pysat._files.Files)
        A class containing the verified available files
    """
    if format_str is None and data_path is not None:
        if (tag == '1min') | (tag == '5min'):
            min_fmt = ''.join(['omni_hro_', tag,
                               '{year:4d}{month:02d}{day:02d}_v01.cdf'])
            files = pysat.Files.from_os(data_path=data_path, format_str=min_fmt)
            # files are by month, just add date to monthly filename for
            # each day of the month. load routine will use date to select out
            # appropriate data
            if not files.empty:
                files.ix[files.index[-1] + pds.DateOffset(months=1) -
                         pds.DateOffset(days=1)] = files.iloc[-1]
                files = files.asfreq('D', 'pad')
                # add the date to the filename
                files = files + '_' + files.index.strftime('%Y-%m-%d')
            return files
        else:
            raise ValueError('Unknown tag')
    elif format_str is None:
        estr = 'A directory must be passed to the loading routine for OMNI HRO'
        raise ValueError (estr)
    else:
        return pysat.Files.from_os(data_path=data_path, format_str=format_str)