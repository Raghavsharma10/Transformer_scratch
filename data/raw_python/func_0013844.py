def append_data(file_strings, file_fmt, tag):
    """ Load the SuperMAG files

    Parameters
    -----------
    file_strings : array-like
        Lists or arrays of strings, where each string contains one file of data
    file_fmt : str
        String denoting file type (ascii or csv)
    tag : string
        String denoting the type of file to load, accepted values are 'indices',
        'all', 'stations', and '' (for only magnetometer data)

    Returns
    -------
    out_string : string
        String with all data, ready for output to a file
        
    """
    # Determine the right appending routine for the file type
    if file_fmt.lower() == "csv":
        return append_csv_data(file_strings)
    else:
        return append_ascii_data(file_strings, tag)