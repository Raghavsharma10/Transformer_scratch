def thresh(dset,p,positive_only=False,prefix=None):
    ''' returns a string containing an inline ``3dcalc`` command that thresholds the
        given dataset at the specified *p*-value, or will create a new dataset if a
        suffix or prefix is given '''
    return available_method('thresh')(dset,p,positive_only,prefix)