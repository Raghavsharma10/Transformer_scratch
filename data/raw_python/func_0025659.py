def getFilterNames(header, filternames=None):
    """
    Returns a comma-separated string of filter names extracted from the input
    header (PyFITS header object).  This function has been hard-coded to
    support the following instruments:

        ACS, WFPC2, STIS

    This function relies on the 'INSTRUME' keyword to define what instrument
    has been used to generate the observation/header.

    The 'filternames' parameter allows the user to provide a list of keyword
    names for their instrument, in the case their instrument is not supported.
    """

    # Define the keyword names for each instrument
    _keydict = {
        'ACS': ['FILTER1', 'FILTER2'],
        'WFPC2': ['FILTNAM1', 'FILTNAM2'],
        'STIS': ['OPT_ELEM', 'FILTER'],
        'NICMOS': ['FILTER', 'FILTER2'],
        'WFC3': ['FILTER', 'FILTER2']
    }

    # Find out what instrument the input header came from, based on the
    # 'INSTRUME' keyword
    if 'INSTRUME' in header:
        instrument = header['INSTRUME']
    else:
        raise ValueError('Header does not contain INSTRUME keyword.')

    # Check to make sure this instrument is supported in _keydict
    if instrument in _keydict:
        _filtlist = _keydict[instrument]
    else:
        _filtlist = filternames

    # At this point, we know what keywords correspond to the filter names
    # in the header.  Now, get the values associated with those keywords.
    # Build a list of all filter name values, with the exception of the
    # blank keywords. Values containing 'CLEAR' or 'N/A' are valid.
    _filter_values = []
    for _key in _filtlist:
        if _key in header:
            _val = header[_key]
        else:
            _val = ''
        if _val.strip() != '':
            _filter_values.append(header[_key])

    # Return the comma-separated list
    return ','.join(_filter_values)