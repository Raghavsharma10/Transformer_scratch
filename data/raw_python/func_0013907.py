def load(fnames, tag=None, sat_id=None):
    """Load Kp index files

    Parameters
    ------------
    fnames : (pandas.Series)
        Series of filenames
    tag : (str or NoneType)
        tag or None (default=None)
    sat_id : (str or NoneType)
        satellite id or None (default=None)

    Returns
    ---------
    data : (pandas.DataFrame)
        Object containing satellite data
    meta : (pysat.Meta)
        Object containing metadata such as column names and units

    Notes
    -----
    Called by pysat. Not intended for direct use by user.
    
    """

    # Kp data stored monthly, need to return data daily
    # the daily date is attached to filename
    # parse off the last date, load month of data, downselect to desired day
    data = pds.DataFrame()
    #set up fixed width format for these files
    colspec = [(0,2),(2,4),(4,6),(7,10),(10,13),(13,16),(16,19),(19,23),(23,26),(26,29),(29,32),(32,50)]
    for filename in fnames:
        # the daily date is attached to filename
        # parse off the last date, load month of data, downselect to desired day
        fname = filename[0:-11]
        date = pysat.datetime.strptime(filename[-10:], '%Y-%m-%d')

        temp = pds.read_fwf(fname, colspecs=colspec, skipfooter=4,header=None, 
                            parse_dates=[[0,1,2]], date_parser=_parse, 
                            index_col='0_1_2')
        idx, = np.where((temp.index >= date) & (temp.index < date+pds.DateOffset(days=1)))
        temp = temp.iloc[idx,:]
        data = pds.concat([data,temp], axis=0)
        
    # drop last column as it has data I don't care about
    data = data.iloc[:,0:-1]
    
    # each column increments UT by three hours
    # produce a single data series that has Kp value monotonically increasing in time
    # with appropriate datetime indices
    s = pds.Series()
    for i in np.arange(8):
        temp = pds.Series(data.iloc[:,i].values, 
                          index=data.index+pds.DateOffset(hours=int(3*i))  )
        #print temp
        s = s.append(temp) 
    s = s.sort_index()
    s.index.name = 'time'
    
    # now, Kp comes in non-user friendly values
    # 2-, 2o, and 2+ relate to 1.6, 2.0, 2.3
    # will convert for user friendliness
    first = np.array([float(x[0]) for x in s])
    flag = np.array([x[1] for x in s])

    ind, = np.where(flag == '+')
    first[ind] += 1./3.
    ind, = np.where(flag == '-')
    first[ind] -= 1./3.
    
    result = pds.DataFrame(first, columns=['kp'], index=s.index)
        
    return result, pysat.Meta()