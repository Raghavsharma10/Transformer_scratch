def get_scan_parameters_index(scan_parameter):
    '''Takes the scan parameter array and creates a scan parameter index labeling the unique scan parameter combinations.
    Parameters
    ----------
    scan_parameter : numpy.ndarray
        The table with the scan parameters.

    Returns
    -------
    numpy.Histogram
    '''
    _, index = np.unique(scan_parameter, return_index=True)
    index = np.sort(index)
    values = np.array(range(0, len(index)), dtype='i4')
    index = np.append(index, len(scan_parameter))
    counts = np.diff(index)
    return np.repeat(values, counts)