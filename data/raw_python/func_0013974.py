def time_shift_to_magnetic_poles(inst):
    """ OMNI data is time-shifted to bow shock. Time shifted again
    to intersections with magnetic pole.

    Parameters
    -----------
    inst : Instrument class object
        Instrument with OMNI HRO data

    Notes
    ---------
    Time shift calculated using distance to bow shock nose (BSN)
    and velocity of solar wind along x-direction.
    
    Warnings
    --------
    Use at own risk.
    
    """
    
    # need to fill in Vx to get an estimate of what is going on
    inst['Vx'] = inst['Vx'].interpolate('nearest')
    inst['Vx'] = inst['Vx'].fillna(method='backfill')
    inst['Vx'] = inst['Vx'].fillna(method='pad')

    inst['BSN_x'] = inst['BSN_x'].interpolate('nearest')
    inst['BSN_x'] = inst['BSN_x'].fillna(method='backfill')
    inst['BSN_x'] = inst['BSN_x'].fillna(method='pad')

    # make sure there are no gaps larger than a minute
    inst.data = inst.data.resample('1T').interpolate('time')

    time_x = inst['BSN_x']*6371.2/-inst['Vx']
    idx, = np.where(np.isnan(time_x))
    if len(idx) > 0:
        print (time_x[idx])
        print (time_x)
    time_x_offset = [pds.DateOffset(seconds = time)
                     for time in time_x.astype(int)]
    new_index=[]
    for i, time in enumerate(time_x_offset):
        new_index.append(inst.data.index[i] + time)
    inst.data.index = new_index
    inst.data = inst.data.sort_index()    
    
    return