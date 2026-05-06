def daily3D(inst, bin1, label1, bin2, label2, bin3, label3, 
            data_label, gate, returnBins=False):
    """3D Daily Occurrence Probability of data_label > gate over a season.
    
    If data_label is greater than gate atleast once per day, 
    then a 100% occurrence probability results. Season delineated by 
    the bounds attached to Instrument object.
    Prob = (# of times with at least one hit)/(# of times in bin)

    Parameters
    ----------
    inst: pysat.Instrument()
        Instrument to use for calculating occurrence probability
    binx: list
        [min, max, number of bins]
    labelx: string 
        name for data product for binx
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
        'prob' for the probability and 'count' for the number of days with any
        data; 'bin_x', 'bin_y', and 'bin_z' are also returned if requested. Note
        that arrays are organized for direct plotting, z,y,x.

    Note
    ----
    Season delineated by the bounds attached to Instrument object.    
            
    """
        
    return _occurrence3D(inst, bin1, label1, bin2, label2, bin3, label3, 
                        data_label, gate, returnBins=returnBins, by_orbit=False)