def detect_gaps(dataframe, timestep, print_all=False, print_max=5, verbose=True):
    """checks if a given dataframe contains gaps and returns the number of gaps

    This funtion checks if a dataframe contains any gaps for a given temporal
    resolution that needs to be specified in seconds. The number of gaps
    detected in the dataframe is returned.

    Args:
        dataframe: A pandas dataframe object with index defined as datetime
        timestep (int): The temporal resolution of the time series in seconds
            (e.g., 86400 for daily values)
        print_all (bool, opt): Lists every gap on the screen
        print_mx (int, opt): The maximum number of gaps listed on the screen in
            order to avoid a decrease in performance if numerous gaps occur
        verbose (bool, opt): Enables/disables output to the screen

    Returns:
        The number of gaps as integer. Negative values indicate errors.
    """
    gcount = 0
    msg_counter = 0
    warning_printed = False
    try:
        n = len(dataframe.index)
    except:
        print('Error: Invalid dataframe.')
        return -1
    for i in range(0, n):
        if(i > 0):
            time_diff = dataframe.index[i] - dataframe.index[i-1]
            if(time_diff.delta/1E9 != timestep):
                gcount += 1
                if print_all or (msg_counter <= print_max - 1):
                    if verbose:
                        print('Warning: Gap in time series found between %s and %s' % (dataframe.index[i-1], dataframe.index[i]))
                    msg_counter += 1
                if msg_counter == print_max and verbose and not warning_printed:
                    print('Waring: Only the first %i gaps have been listed. Try to increase print_max parameter to show more details.' % msg_counter)
                    warning_printed = True
    if verbose:
        print('%i gaps found in total.' % (gcount))
    return gcount