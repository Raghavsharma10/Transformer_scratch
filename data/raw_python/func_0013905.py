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

    Note
    ----
    Called by pysat. Not intended for direct use by user.
    
    """

    data = pds.DataFrame()
    
    for filename in fnames:
        # need to remove date appended to dst filename
        fname = filename[0:-11]
        #f = open(fname)
        with open(fname) as f:
            lines = f.readlines()
            idx = 0
            # check if all lines are good
            max_lines=0
            for line in lines:
                if len(line) > 1:
                    max_lines+=1
            yr = np.zeros(max_lines*24, dtype=int)
            mo = np.zeros(max_lines*24, dtype=int)
            day = np.zeros(max_lines*24, dtype=int)
            ut = np.zeros(max_lines*24, dtype=int)
            dst = np.zeros(max_lines*24, dtype=int)
            for line in lines:
                if len(line) > 1:
                    temp_year = int(line[14:16] + line[3:5]) 
                    if temp_year > 57:
                        temp_year += 1900
                    else:
                        temp_year += 2000
                        
                    yr[idx:idx+24] = temp_year
                    mo[idx:idx+24] = int(line[5:7])
                    day[idx:idx+24] = int(line[8:10])
                    ut[idx:idx+24] = np.arange(24)
                    temp = line.strip()[20:-4]
                    temp2 = [temp[4*i:4*(i+1)] for i in np.arange(24)]
                    dst[idx:idx+24] = temp2
                    idx += 24   
    
            #f.close()
    
            
            start = pds.datetime(yr[0], mo[0], day[0], ut[0])
            stop = pds.datetime(yr[-1], mo[-1], day[-1], ut[-1])
            dates = pds.date_range(start, stop, freq='H') 
            
            new_data = pds.DataFrame(dst, index=dates, columns=['dst'])
            # pull out specific day 
            new_date = pysat.datetime.strptime(filename[-10:], '%Y-%m-%d')
            idx, = np.where((new_data.index >= new_date) & (new_data.index < new_date+pds.DateOffset(days=1)))
            new_data = new_data.iloc[idx,:]
            # add specific day to all data loaded for filenames
            data = pds.concat([data, new_data], axis=0) 
        
    return data, pysat.Meta()