def list_files(tag=None, sat_id=None, data_path=None, format_str=None,
               supported_tags=None, fake_daily_files_from_monthly=False,
               two_digit_year_break=None):
    """Return a Pandas Series of every file for chosen satellite data.
    
    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are <tag strings>. (default=None)
    sat_id : (string or NoneType)
        Specifies the satellite ID for a constellation.  Not used.
        (default=None)
    data_path : (string or NoneType)
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)
    format_str : (string or NoneType)
        User specified file format.  If None is specified, the default
        formats associated with the supplied tags are used. (default=None)
    supported_tags : (dict or NoneType)
        keys are tags supported by list_files routine. Values are the
        default format_str values for key. (default=None)
    fake_daily_files_from_monthly : bool
        Some CDAWeb instrument data files are stored by month, interfering
        with pysat's functionality of loading by day. This flag, when true,
        appends daily dates to monthly files internally. These dates are
        used by load routine in this module to provide data by day. 

    Returns
    --------
    pysat.Files.from_os : (pysat._files.Files)
        A class containing the verified available files

    Examples
    --------
    :: 

        fname = 'cnofs_vefi_bfield_1sec_{year:04d}{month:02d}{day:02d}_v05.cdf'
        supported_tags = {'dc_b':fname}
        list_files = functools.partial(nasa_cdaweb_methods.list_files, 
                                       supported_tags=supported_tags)

        ivm_fname = 'cnofs_cindi_ivm_500ms_{year:4d}{month:02d}{day:02d}_v01.cdf'
        supported_tags = {'':ivm_fname}
        list_files = functools.partial(cdw.list_files, 
                                       supported_tags=supported_tags)
    
    """

    if data_path is not None:
        if format_str is None:
                try:
                    format_str = supported_tags[sat_id][tag]
                except KeyError:
                    raise ValueError('Unknown tag')
        out = pysat.Files.from_os(data_path=data_path, 
                                  format_str=format_str)

        if (not out.empty) and fake_daily_files_from_monthly:
            out.ix[out.index[-1] + pds.DateOffset(months=1) -
                                   pds.DateOffset(days=1)] = out.iloc[-1]  
            out = out.asfreq('D', 'pad')
            out = out + '_' + out.index.strftime('%Y-%m-%d')  
            return out

        return out
    else:
        estr = 'A directory must be passed to the loading routine for <Instrument Code>'
        raise ValueError (estr)