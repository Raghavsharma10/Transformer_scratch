def get_scan_parameter(meta_data_array, unique=True):
    '''Takes the numpy meta data array and returns the different scan parameter settings and the name aligned in a dictionary

    Parameters
    ----------
    meta_data_array : numpy.ndarray
    unique: boolean
        If true only unique values for each scan parameter are returned

    Returns
    -------
    python.dict{string, numpy.Histogram}:
        A dictionary with the scan parameter name/values pairs
    '''

    try:
        last_not_parameter_column = meta_data_array.dtype.names.index('error_code')  # for interpreted meta_data
    except ValueError:
        last_not_parameter_column = meta_data_array.dtype.names.index('error')  # for raw data file meta_data
    if last_not_parameter_column == len(meta_data_array.dtype.names) - 1:  # no meta_data found
        return
    scan_parameters = collections.OrderedDict()
    for scan_par_name in meta_data_array.dtype.names[4:]:  # scan parameters are in columns 5 (= index 4) and above
        scan_parameters[scan_par_name] = np.unique(meta_data_array[scan_par_name]) if unique else meta_data_array[scan_par_name]
    return scan_parameters