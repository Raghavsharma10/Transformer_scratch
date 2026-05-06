def list_files(tag=None, sat_id=None, data_path=None, format_str=None):
    """Produce a list of files corresponding to format_str located at data_path.

    This routine is invoked by pysat and is not intended for direct use by the end user.
    
    Multiple data levels may be supported via the 'tag' and 'sat_id' input strings.

    Parameters
    ----------
    tag : string ('')
        tag name used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.
    sat_id : string ('')
        Satellite ID used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.
    data_path : string
        Full path to directory containing files to be loaded. This
        is provided by pysat. The user may specify their own data path
        at Instrument instantiation and it will appear here.
    format_str : string (None)
        String template used to parse the datasets filenames. If a user
        supplies a template string at Instrument instantiation
        then it will appear here, otherwise defaults to None.
    
    Returns
    -------
    pandas.Series
        Series of filename strings, including the path, indexed by datetime.
    
    Examples
    --------
    ::
    
        If a filename is SPORT_L2_IVM_2019-01-01_v01r0000.NC then the template
        is 'SPORT_L2_IVM_{year:04d}-{month:02d}-{day:02d}_v{version:02d}r{revision:04d}.NC'
    
    Note
    ----
    The returned Series should not have any duplicate datetimes. If there are
    multiple versions of a file the most recent version should be kept and the rest
    discarded. This routine uses the pysat.Files.from_os constructor, thus
    the returned files are up to pysat specifications.
    
    Normally the format_str for each supported tag and sat_id is defined within this routine.
    However, as this is a generic routine, those definitions can't be made here. This method
    could be used in an instrument specific module where the list_files routine in the
    new package defines the format_str based upon inputs, then calls this routine passing
    both data_path and format_str.
    
    Alternately, the list_files routine in nasa_cdaweb_methods may also be used and has
    more built in functionality. Supported tages and format strings may be defined
    within the new instrument module and passed as arguments to nasa_cdaweb_methods.list_files .
    For an example on using this routine, see pysat/instrument/cnofs_ivm.py or cnofs_vefi, cnofs_plp,
    omni_hro, timed_see, etc.
    
    """
    
    return pysat.Files.from_os(data_path=data_path, format_str=format_str)