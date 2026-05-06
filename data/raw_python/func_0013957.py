def create_datetime_index(year=None, month=None, day=None, uts=None):
    """Create a timeseries index using supplied year, month, day, and ut in
    seconds.

    Parameters
    ----------
        year : array_like of ints 
        month : array_like of ints or None
        day : array_like of ints
            for day (default) or day of year (use month=None)
        uts : array_like of floats

    Returns
    -------
        Pandas timeseries index.

    Note
    ----
    Leap seconds have no meaning here.
    """
    # need a timeseries index for storing satellite data in pandas but
    # creating a datetime object for everything is too slow
    # so I calculate the number of nanoseconds elapsed since first sample, 
    # and create timeseries index from that. 
    # Factor of 20 improvement compared to previous method,
    # which itself was an order of magnitude faster than datetime.
 
    #get list of unique year, and month
    if not hasattr(year, '__iter__'):
        raise ValueError('Must provide an iterable for all inputs.')
    if len(year) == 0:
        raise ValueError('Length of array must be larger than 0.')
    year = year.astype(int)
    if month is None:
        month = np.ones(len(year), dtype=int)
    else:
        month = month.astype(int)

    if uts is None:
        uts = np.zeros(len(year))
    if day is None:
        day = np.ones(len(year))
    day = day.astype(int)
    # track changes in seconds
    uts_del = uts.copy().astype(float)
    # determine where there are changes in year and month that need to be 
    # accounted for    
    _,idx = np.unique(year*100.+month, return_index=True)
    # create another index array for faster algorithm below
    idx2 = np.hstack((idx,len(year)+1))   
    # computes UTC seconds offset for each unique set of year and month
    for _idx,_idx2 in zip(idx[1:],idx2[2:]):
        temp = (datetime(year[_idx],month[_idx],1)
                - datetime(year[0],month[0],1))
        uts_del[_idx:_idx2] += temp.total_seconds()

    # add in UTC seconds for days, ignores existence of leap seconds
    uts_del += (day-1)*86400
    # add in seconds since unix epoch to first day
    uts_del += (datetime(year[0],month[0],1)-datetime(1970,1,1)).total_seconds()
    # going to use routine that defaults to nanseconds for epoch
    uts_del *= 1E9
    return pds.to_datetime(uts_del)