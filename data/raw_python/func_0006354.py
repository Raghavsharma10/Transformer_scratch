def get_meta_data_index_at_scan_parameter(meta_data_array, scan_parameter_name):
    '''Takes the analyzed meta_data table and returns the indices where the scan parameter changes

    Parameters
    ----------
    meta_data_array : numpy.recordarray
    scan_parameter_name : string

    Returns
    -------
    numpy.ndarray:
        first dimension: scan parameter value
        second dimension: index where scan parameter value was used first
    '''
    scan_parameter_values = meta_data_array[scan_parameter_name]
    diff = np.concatenate(([1], np.diff(scan_parameter_values)))
    idx = np.concatenate((np.where(diff)[0], [len(scan_parameter_values)]))
    index = np.empty(len(idx) - 1, dtype={'names': [scan_parameter_name, 'index'], 'formats': ['u4', 'u4']})
    index[scan_parameter_name] = scan_parameter_values[idx[:-1]]
    index['index'] = idx[:-1]
    return index