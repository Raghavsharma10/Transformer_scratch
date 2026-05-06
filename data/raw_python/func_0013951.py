def by_orbit2D(inst, bin1, label1, bin2, label2, data_label, gate, returnBins=False):
    """2D Occurrence Probability of data_label orbit-by-orbit over a season.
    
    If data_label is greater than gate atleast once per orbit, then a 
    100% occurrence probability results. Season delineated by the bounds
    attached to Instrument object.
    Prob = (# of times with at least one hit)/(# of times in bin)
    
    Parameters
    ----------
    inst: pysat.Instrument()
        Instrument to use for calculating occurrence probability
    binx: list
        [min value, max value, number of bins]
    labelx: string 
        identifies data product for binx
    data_label: list of strings 
        identifies data product(s) to calculate occurrence probability
    gate: list of values 
        values that data_label must achieve to be counted as an occurrence
    returnBins: Boolean
        if True, return arrays with values of bin edges, useful for pcolor

    Returns
    -------
    occur_prob : dictionary
        A dict of dicts indexed by data_label. Each entry is dict with entries
        'prob' for the probability and 'count' for the number of orbits with any
        data; 'bin_x' and 'bin_y' are also returned if requested. Note that arrays
        are organized for direct plotting, y values along rows, x along columns.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.
    
    """
    
    return _occurrence2D(inst, bin1, label1, bin2, label2, data_label, gate,
                        by_orbit=True, returnBins=returnBins)