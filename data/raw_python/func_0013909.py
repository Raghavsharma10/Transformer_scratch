def filter_geoquiet(sat, maxKp=None, filterTime=None, kpData=None, kp_inst=None):
    """Filters pysat.Instrument data for given time after Kp drops below gate.
    
    Loads Kp data for the same timeframe covered by sat and sets sat.data to
    NaN for times when Kp > maxKp and for filterTime after Kp drops below maxKp.
    
    Parameters
    ----------
    sat : pysat.Instrument
        Instrument to be filtered
    maxKp : float
        Maximum Kp value allowed. Kp values above this trigger
        sat.data filtering.
    filterTime : int
        Number of hours to filter data after Kp drops below maxKp
    kpData : pysat.Instrument (optional)
        Kp pysat.Instrument object with data already loaded
    kp_inst : pysat.Instrument (optional)
        Kp pysat.Instrument object ready to load Kp data.Overrides kpData.
        
    Returns
    -------
    None : NoneType
        sat Instrument object modified in place
        
    """
    if kp_inst is not None:
        kp_inst.load(date=sat.date, verifyPad=True)
        kpData = kp_inst
    elif kpData is None:
        kp = pysat.Instrument('sw', 'kp', pad=pds.DateOffset(days=1))
        kp.load(date=sat.date, verifyPad=True)
        kpData = kp
        
    
    if maxKp is None:
        maxKp = 3+ 1./3.
        
    if filterTime is None:
        filterTime = 24
        
    # now the defaults are ensured, let's do some filtering
    # date of satellite data
    date = sat.date
    selData = kpData[date-pds.DateOffset(days=1):date+pds.DateOffset(days=1)]
    ind, = np.where(selData['kp'] >= maxKp)
    for lind in ind:
        sat.data[selData.index[lind]:(selData.index[lind]+pds.DateOffset(hours=filterTime) )] = np.NaN
        sat.data = sat.data.dropna(axis=0, how='all')

    return