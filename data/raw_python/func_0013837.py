def list_files(tag='', sat_id=None, data_path=None, format_str=None):
    """Return a Pandas Series of every file for chosen SuperMAG data

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are 'indices', 'all',
        'stations', and '' (for just magnetometer measurements). (default='')
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
        file_base = 'supermag_magnetometer'
        if tag == "indices" or tag == "all":
            file_base += '_all' # Can't just download indices

            if tag == "indices":
                psplit = path.split(data_path[:-1])
                data_path = path.join(psplit[0], "all", "")

        if tag == "stations":
            min_fmt = '_'.join([file_base, '{year:4d}.???'])
            doff = pds.DateOffset(years=1)
        else:
            min_fmt = '_'.join([file_base, '{year:4d}{month:02d}{day:02d}.???'])
            doff = pds.DateOffset(days=1)
        files = pysat.Files.from_os(data_path=data_path, format_str=min_fmt)

        # station files are once per year but we need to
        # create the illusion there is a file per year        
        if not files.empty:
            files = files.sort_index()

            if tag == "stations":
                orig_files = files.copy()
                new_files = []
                # Assigns the validity of each station file to be 1 year
                for orig in orig_files.iteritems():
                    files.ix[orig[0] + doff - pds.DateOffset(days=1)] = orig[1]
                    files = files.sort_index()
                    new_files.append(files.ix[orig[0]: orig[0] + doff - \
                            pds.DateOffset(days=1)].asfreq('D', method='pad'))
                files = pds.concat(new_files)

                files = files.dropna()
                files = files.sort_index()
            # add the date to the filename
            files = files + '_' + files.index.strftime('%Y-%m-%d')
        return files
    elif format_str is None:
        estr = 'A directory must be passed to the loading routine for SuperMAG'
        raise ValueError (estr)
    else:
        return pysat.Files.from_os(data_path=data_path, format_str=format_str)