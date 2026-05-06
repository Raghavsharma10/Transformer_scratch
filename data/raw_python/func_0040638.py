def cutoff_filename(prefix, suffix, input_str):
    """
    Cuts off the start and end of a string, as specified by 2 parameters

    Parameters
    ----------
    prefix : string, if input_str starts with prefix, will cut off prefix
    suffix : string, if input_str end with suffix, will cut off suffix
    input_str : the string to be processed

    Returns
    -------
    A string, from which the start and end have been cut
    """
    if prefix is not '':
        if input_str.startswith(prefix):
            input_str = input_str[len(prefix):]
    if suffix is not '':
        if input_str.endswith(suffix):
            input_str = input_str[:-len(suffix)]
    return input_str