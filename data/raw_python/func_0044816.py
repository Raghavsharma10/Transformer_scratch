def show_stat_base(count_value, max_count_value, prepend, speed, tet, ttg, width, **kwargs):
    """A function that formats the progress information

    This function will be called periodically for each progress that is monitored.
    Overwrite this function in a subclass to implement a specific formating of the progress information

    :param count_value:      a number holding the current state
    :param max_count_value:  should be the largest number `count_value` can reach
    :param prepend:          additional text for each progress
    :param speed:            the speed estimation
    :param tet:              the total elapsed time
    :param ttg:              the time to go
    :param width:            the width for the progressbar, when set to `"auto"` this function
        should try to detect the width available
    :type width:             int or "auto"
    """
    raise NotImplementedError