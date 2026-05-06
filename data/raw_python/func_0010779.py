def drop_incomplete_days(dataframe, shift=0):
    """truncates a given dataframe to full days only

    This funtion truncates a given pandas dataframe (time series) to full days
    only, thus dropping leading and tailing hours of incomplete days. Please
    note that this methodology only applies to hourly time series.

    Args:
        dataframe: A pandas dataframe object with index defined as datetime
        shift (unsigned int, opt): First hour of daily recordings. For daily
            recordings of precipitation gages, 8 would be the first hour of
            the subsequent day of recordings since daily totals are
            usually recorded at 7. Omit defining this parameter if you intend
            to pertain recordings to 0-23h.

    Returns:
        A dataframe with full days only.
    """
    dropped = 0
    if shift > 23 or shift < 0:
        print("Invalid shift parameter setting! Using defaults.")
        shift = 0
    first = shift
    last = first - 1
    if last < 0:
        last += 24
    try:
        # todo: move this checks to a separate function
        n = len(dataframe.index)
    except:
        print('Error: Invalid dataframe.')
        return dataframe
    
    delete = list()  
    
    # drop heading lines if required
    for i in range(0, n):
        if dataframe.index.hour[i] == first and dataframe.index.minute[i] == 0:
            break
        else:
            delete.append(i)
            dropped += 1

    # drop tailing lines if required
    for i in range(n-1, 0, -1):
        if dataframe.index.hour[i] == last and dataframe.index.minute[i] == 0:
            break
        else:
            delete.append(i)
            dropped += 1
    # print("The following rows have been dropped (%i in total):" % dropped)
    # print(delete)
    return dataframe.drop(dataframe.index[[delete]])