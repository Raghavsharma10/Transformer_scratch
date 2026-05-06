def clean(self):
    """Routine to return C/NOFS IVM data cleaned to the specified level

    Parameters
    -----------
    inst : (pysat.Instrument)
        Instrument class object, whose attribute clean_level is used to return
        the desired level of data selectivity.

    Returns
    --------
    Void : (NoneType)
        data in inst is modified in-place.

    Notes
    --------
    Supports 'clean', 'dusty', 'dirty'
    
    """

    # cleans cindi data
    if self.clean_level == 'clean':
        # choose areas below 550km
        # self.data = self.data[self.data.alt <= 550]
        idx, = np.where(self.data.altitude <= 550)
        self.data = self[idx,:]
    
    # make sure all -999999 values are NaN
    self.data.replace(-999999., np.nan, inplace=True)

    if (self.clean_level == 'clean') | (self.clean_level == 'dusty'):
        try:
            idx, = np.where(np.abs(self.data.ionVelmeridional) < 10000.)
            self.data = self[idx,:]
        except AttributeError:
            pass
        
        if self.clean_level == 'dusty':
            # take out all values where RPA data quality is > 1
            idx, = np.where(self.data.RPAflag <= 1)
            self.data = self[idx,:]
            # IDM quality flags
            self.data = self.data[ (self.data.driftMeterflag<= 3) ]
        else:
            # take out all values where RPA data quality is > 0
            idx, = np.where(self.data.RPAflag <= 0)
            self.data = self[idx,:] 
            # IDM quality flags
            self.data = self.data[ (self.data.driftMeterflag<= 0) ]
    if self.clean_level == 'dirty':
        # take out all values where RPA data quality is > 4
        idx, = np.where(self.data.RPAflag <= 4)
        self.data = self[idx,:]
        # IDM quality flags
        self.data = self.data[ (self.data.driftMeterflag<= 6) ]
        
    # basic quality check on drifts and don't let UTS go above 86400.
    idx, = np.where(self.data.time <= 86400.)
    self.data = self[idx,:]
    
    # make sure MLT is between 0 and 24
    idx, = np.where((self.data.mlt >= 0) & (self.data.mlt <= 24.))
    self.data = self[idx,:]
    return