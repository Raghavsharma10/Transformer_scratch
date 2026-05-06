def list_files(tag=None, sat_id=None, data_path=None, format_str=None):
    """Return a Pandas Series of every file for chosen satellite data

    Parameters
    -----------
    tag : (string or NoneType)
        Denotes type of file to load.  Accepted types are '' and 'ascii'.
        If '' is specified, the primary data type (ascii) is loaded.
        (default=None)
    sat_id : (string or NoneType)
        Specifies the satellite ID for a constellation.  Not used.
        (default=None)
    data_path : (string or NoneType)
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)
    format_str : (NoneType)
        User specified file format not supported. (default=None)

    Returns
    --------
    pysat.Files.from_os : (pysat._files.Files)
        A class containing the verified available files
    """
    import sys
    #if tag == 'ionprf':
    #    # from_os constructor currently doesn't work because of the variable 
    #    # filename components at the end of each string.....
    #    ion_fmt = '*/ionPrf_*.{year:04d}.{day:03d}.{hour:02d}.{min:02d}*_nc'
    #    return pysat.Files.from_os(dir_path=os.path.join('cosmic', 'ionprf'),
    #                               format_str=ion_fmt)
    estr = 'Building a list of COSMIC files, which can possibly take time. '
    estr = '{:s}~1s per 100K files'.format(estr)
    print(estr)
    sys.stdout.flush()

    # number of files may be large
    # only select file that are the cosmic data files and end with _nc
    cosmicFiles = glob.glob(os.path.join(data_path, '*/*_nc'))
    # need to get date and time from filename to generate index
    num = len(cosmicFiles) 
    if num != 0:
        print('Estimated time:', num*1.E-5,'seconds')
        sys.stdout.flush()
        # preallocate lists
        year=[None]*num; days=[None]*num; hours=[None]*num; 
        minutes=[None]*num; microseconds=[None]*num;
        for i,f in enumerate(cosmicFiles):
            f2 = f.split('.')
            year[i]=f2[-6]
            days[i]=f2[-5]
            hours[i]=f2[-4]
            minutes[i]=f2[-3]
            microseconds[i]=i
    
        year=np.array(year).astype(int)
        days=np.array(days).astype(int)
        uts=np.array(hours).astype(int)*3600.+np.array(minutes).astype(int)*60.
        # adding microseconds to ensure each time is unique, not allowed to
        # pass 1.E-3 s
        uts+=np.mod(np.array(microseconds).astype(int)*4, 8000)*1.E-5
        index = pysat.utils.create_datetime_index(year=year, day=days, uts=uts)
        file_list = pysat.Series(cosmicFiles, index=index)
        return file_list
    else:
        print('Found no files, check your path or download them.')
        return pysat.Series(None)