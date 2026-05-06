def season_date_range(start, stop, freq='D'):
    """
    Return array of datetime objects using input frequency from start to stop

    Supports single datetime object or list, tuple, ndarray of start and 
    stop dates.

    freq codes correspond to pandas date_range codes, D daily, M monthly,
    S secondly
    """

    if hasattr(start, '__iter__'):  
        # missing check for datetime
        season = pds.date_range(start[0], stop[0], freq=freq)
        for (sta,stp) in zip(start[1:], stop[1:]):
            season = season.append(pds.date_range(sta, stp, freq=freq))
    else:
        season = pds.date_range(start, stop, freq=freq)
    return season