def add_mag_drifts(inst):
    """Adds ion drifts in magnetic coordinates using ion drifts in S/C coordinates
    along with pre-calculated unit vectors for magnetic coordinates.
    
    Note
    ----
        Requires ion drifts under labels 'iv_*' where * = (x,y,z) along with
        unit vectors labels 'unit_zonal_*', 'unit_fa_*', and 'unit_mer_*',
        where the unit vectors are expressed in S/C coordinates. These
        vectors are calculated by add_mag_drift_unit_vectors.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object will be modified to include new ion drift magnitudes
        
    Returns
    -------
    None
        Instrument object modified in place
    
    """
    
    inst['iv_zon'] = {'data':inst['unit_zon_x'] * inst['iv_x'] + inst['unit_zon_y']*inst['iv_y'] + inst['unit_zon_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Zonal ion velocity',
                      'notes':('Ion velocity relative to co-rotation along zonal '
                               'direction, normal to meridional plane. Positive east. '
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label': 'Zonal Ion Velocity',
                      'axis': 'Zonal Ion Velocity',
                      'desc': 'Zonal ion velocity',
                      'scale': 'Linear',
                      'value_min':-500., 
                      'value_max':500.}
                      
    inst['iv_fa'] = {'data':inst['unit_fa_x'] * inst['iv_x'] + inst['unit_fa_y'] * inst['iv_y'] + inst['unit_fa_z'] * inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Field-Aligned ion velocity',
                      'notes':('Ion velocity relative to co-rotation along magnetic field line. Positive along the field. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Field-Aligned Ion Velocity',
                      'axis':'Field-Aligned Ion Velocity',
                      'desc':'Field-Aligned Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}

    inst['iv_mer'] = {'data':inst['unit_mer_x'] * inst['iv_x'] + inst['unit_mer_y']*inst['iv_y'] + inst['unit_mer_z']*inst['iv_z'],
                      'units':'m/s',
                      'long_name':'Meridional ion velocity',
                      'notes':('Velocity along meridional direction, perpendicular '
                               'to field and within meridional plane. Positive is up at magnetic equator. ',
                               'Velocity obtained using ion velocities relative '
                               'to co-rotation in the instrument frame along '
                               'with the corresponding unit vectors expressed in '
                               'the instrument frame. '),
                      'label':'Meridional Ion Velocity',
                      'axis':'Meridional Ion Velocity',
                      'desc':'Meridional Ion Velocity',
                      'scale':'Linear',
                      'value_min':-500., 
                      'value_max':500.}
    
    return