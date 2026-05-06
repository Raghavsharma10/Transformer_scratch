def list_files(tag='', sat_id=None, data_path=None, format_str=None):
    """Return a Pandas Series of every file for chosen satellite data

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are '' and 'ascii'.
        If '' is specified, the primary data type (ascii) is loaded.
        (default='')
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

    if format_str is None and tag is not None:
        if tag == '' or tag == 'ascii':
            ascii_fmt = 'Density_3deg_{year:02d}_{doy:03d}.ascii'
            return pysat.Files.from_os(data_path=data_path,
                                       format_str=ascii_fmt)
        else:
            raise ValueError('Unrecognized tag name for CHAMP STAR')
    elif format_str is None:
        estr = 'A tag name must be passed to the loading routine for CHAMP'
        raise ValueError(estr)
    else:
        return pysat.Files.from_os(data_path=data_path, format_str=format_str)