def add_footpoint_and_equatorial_drifts(inst, equ_mer_scalar='equ_mer_drifts_scalar',
                                              equ_zonal_scalar='equ_zon_drifts_scalar',
                                              north_mer_scalar='north_footpoint_mer_drifts_scalar',
                                              north_zon_scalar='north_footpoint_zon_drifts_scalar',
                                              south_mer_scalar='south_footpoint_mer_drifts_scalar',
                                              south_zon_scalar='south_footpoint_zon_drifts_scalar',
                                              mer_drift='iv_mer',
                                              zon_drift='iv_zon'):
    """Translates geomagnetic ion velocities to those at footpoints and magnetic equator.
    Note
    ----
        Presumes scalar values for mapping ion velocities are already in the inst, labeled
        by north_footpoint_zon_drifts_scalar, north_footpoint_mer_drifts_scalar,
        equ_mer_drifts_scalar, equ_zon_drifts_scalar.
    
        Also presumes that ion motions in the geomagnetic system are present and labeled
        as 'iv_mer' and 'iv_zon' for meridional and zonal ion motions.
        
        This naming scheme is used by the other pysat oriented routines
        in this package.
    
    Parameters
    ----------
    inst : pysat.Instrument
    equ_mer_scalar : string
        Label used to identify equatorial scalar for meridional ion drift
    equ_zon_scalar : string
        Label used to identify equatorial scalar for zonal ion drift
    north_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    north_zon_scalar : string
        Label used to identify northern footpoint scalar for zonal ion drift
    south_mer_scalar : string
        Label used to identify northern footpoint scalar for meridional ion drift
    south_zon_scalar : string
        Label used to identify southern footpoint scalar for zonal ion drift
    mer_drift : string
        Label used to identify meridional ion drifts within inst
    zon_drift : string
        Label used to identify zonal ion drifts within inst
        
    Returns
    -------
    None
        Modifies pysat.Instrument object in place. Drifts mapped to the magnetic equator
        are labeled 'equ_mer_drift' and 'equ_zon_drift'. Mappings to the northern
        and southern footpoints are labeled 'south_footpoint_mer_drift' and
        'south_footpoint_zon_drift'. Similarly for the northern hemisphere.
    """

    inst['equ_mer_drift'] = {'data' : inst[equ_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Equatorial meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'magnetic equator. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Meridional Ion Velocity',
                            'axis':'Equatorial Meridional Ion Velocity',
                            'desc':'Equatorial Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['equ_zon_drift'] = {'data' : inst[equ_zonal_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Equatorial zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'magnetic equator. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic equator. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Equatorial Zonal Ion Velocity',
                            'axis':'Equatorial Zonal Ion Velocity',
                            'desc':'Equatorial Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_mer_drift'] = {'data' : inst[south_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Southern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'southern footpoint. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Southern Meridional Ion Velocity',
                            'axis':'Southern Meridional Ion Velocity',
                            'desc':'Southern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['south_footpoint_zon_drift'] = {'data':inst[south_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Southern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'southern footpoint. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the southern footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Southern Zonal Ion Velocity',
                            'axis':'Southern Zonal Ion Velocity',
                            'desc':'Southern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_mer_drift'] = {'data':inst[north_mer_scalar]*inst[mer_drift],
                            'units':'m/s',
                            'long_name':'Northern meridional ion velocity',
                            'notes':('Velocity along meridional direction, perpendicular '
                                    'to field and within meridional plane, scaled to '
                                    'northern footpoint. Positive is up at magnetic equator. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the magnetic footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Northern Meridional Ion Velocity',
                            'axis':'Northern Meridional Ion Velocity',
                            'desc':'Northern Meridional Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}

    inst['north_footpoint_zon_drift'] = {'data':inst[north_zon_scalar]*inst[zon_drift],
                            'units':'m/s',
                            'long_name':'Northern zonal ion velocity',
                            'notes':('Velocity along zonal direction, perpendicular '
                                    'to field and the meridional plane, scaled to '
                                    'northern footpoint. Positive is generally eastward. '
                                    'Velocity obtained using ion velocities relative '
                                    'to co-rotation in the instrument frame along '
                                    'with the corresponding unit vectors expressed in '
                                    'the instrument frame. Field-line mapping and '
                                    'the assumption of equi-potential field lines '
                                    'is used to translate the locally measured ion '
                                    'motion to the northern footpoint. The mapping '
                                    'is used to determine the change in magnetic '
                                    'field line distance, which, under assumption of '
                                    'equipotential field lines, in turn alters '
                                    'the electric field at that location (E=V/d). '),
                            'label':'Northern Zonal Ion Velocity',
                            'axis':'Northern Zonal Ion Velocity',
                            'desc':'Northern Zonal Ion Velocity',
                            'scale':'Linear',
                            'value_min':-500., 
                            'value_max':500.}