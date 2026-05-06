def get_unique_scan_parameter_combinations(meta_data_array, scan_parameters=None, scan_parameter_columns_only=False):
    '''Takes the numpy meta data array and returns the first rows with unique combinations of different scan parameter values for selected scan parameters.
        If selected columns only is true, the returned histogram only contains the selected columns.

    Parameters
    ----------
    meta_data_array : numpy.ndarray
    scan_parameters : list of string, None
        Scan parameter names taken. If None all are used.
    selected_columns_only : bool

    Returns
    -------
    numpy.Histogram
    '''

    try:
        last_not_parameter_column = meta_data_array.dtype.names.index('error_code')  # for interpreted meta_data
    except ValueError:
        last_not_parameter_column = meta_data_array.dtype.names.index('error')  # for raw data file meta_data
    if last_not_parameter_column == len(meta_data_array.dtype.names) - 1:  # no meta_data found
        return
    if scan_parameters is None:
        return unique_row(meta_data_array, use_columns=range(4, len(meta_data_array.dtype.names)), selected_columns_only=scan_parameter_columns_only)
    else:
        use_columns = []
        for scan_parameter in scan_parameters:
            try:
                use_columns.append(meta_data_array.dtype.names.index(scan_parameter))
            except ValueError:
                logging.error('No scan parameter ' + scan_parameter + ' found')
                raise RuntimeError('Scan parameter not found')
        return unique_row(meta_data_array, use_columns=use_columns, selected_columns_only=scan_parameter_columns_only)