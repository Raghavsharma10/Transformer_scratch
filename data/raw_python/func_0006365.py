def get_scan_parameters_table_from_meta_data(meta_data_array, scan_parameters=None):
    '''Takes the meta data array and returns the scan parameter values as a view of a numpy array only containing the parameter data .
    Parameters
    ----------
    meta_data_array : numpy.ndarray
        The array with the scan parameters.
    scan_parameters : list of strings
        The name of the scan parameters to take. If none all are used.

    Returns
    -------
    numpy.Histogram
    '''

    if scan_parameters is None:
        try:
            last_not_parameter_column = meta_data_array.dtype.names.index('error_code')  # for interpreted meta_data
        except ValueError:
            return
        if last_not_parameter_column == len(meta_data_array.dtype.names) - 1:  # no meta_data found
            return
        # http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
        scan_par_data = {name: meta_data_array.dtype.fields[name] for name in meta_data_array.dtype.names[last_not_parameter_column + 1:]}
    else:
        scan_par_data = collections.OrderedDict()
        for name in scan_parameters:
            scan_par_data[name] = meta_data_array.dtype.fields[name]

    return np.ndarray(meta_data_array.shape, np.dtype(scan_par_data), meta_data_array, 0, meta_data_array.strides)