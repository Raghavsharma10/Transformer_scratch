def clean(self):
    """Routine to return DMSP IVM data cleaned to the specified level

    'Clean' enforces that both RPA and DM flags are <= 1
    'Dusty' <= 2
    'Dirty' <= 3
    'None' None
    
    Routine is called by pysat, and not by the end user directly.
    
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
    
    if self.clean_level == 'clean':
        idx, = np.where((self['rpa_flag_ut'] <= 1) & (self['idm_flag_ut'] <= 1))
    elif self.clean_level == 'dusty':
        idx, = np.where((self['rpa_flag_ut'] <= 2) & (self['idm_flag_ut'] <= 2))
    elif self.clean_level == 'dirty':
        idx, = np.where((self['rpa_flag_ut'] <= 3) & (self['idm_flag_ut'] <= 3))
    else:
        idx = []
        
    # downselect data based upon cleaning conditions above
    self.data = self[idx]
        
    return