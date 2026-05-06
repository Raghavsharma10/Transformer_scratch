def get_compression_filter(byte_counts):
    """Determine whether or not to use a compression on the array stored in
        a hierarchical data format, and which compression library to use to that purpose.
        Compression reduces the HDF5 file size and also helps improving I/O efficiency
        for large datasets.
    
    Parameters
    ----------
    byte_counts : int
    
    Returns
    -------
    FILTERS : instance of the tables.Filters class
    """

    assert isinstance(byte_counts, numbers.Integral) and byte_counts > 0
    
    if 2 * byte_counts > 1000 * memory()['free']:
        try:
            FILTERS = tables.filters(complevel = 5, complib = 'blosc', 
                                     shuffle = True, least_significant_digit = 6)
        except tables.FiltersWarning:
            FILTERS = tables.filters(complevel = 5, complib = 'lzo', 
                                     shuffle = True, least_significant_digit = 6)   
    else:
        FILTERS = None

    return FILTERS