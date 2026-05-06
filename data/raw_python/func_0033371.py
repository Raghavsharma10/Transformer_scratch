def add_mag_drift_unit_vectors(inst, max_steps=40000, step_size=10.):
    """Add unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in S/C coordinates.
    
    Interally, routine calls add_mag_drift_unit_vectors_ecef. 
    See function for input parameter description.
    Requires the orientation of the S/C basis vectors in ECEF using naming,
    'sc_xhat_x' where *hat (*=x,y,z) is the S/C basis vector and _* (*=x,y,z)
    is the ECEF direction. 
    
    Parameters
    ----------
    inst : pysat.Instrument object
        Instrument object to be modified
    max_steps : int
        Maximum number of steps taken for field line integration
    step_size : float
        Maximum step size (km) allowed for field line tracer
    
    Returns
    -------
    None
        Modifies instrument object in place. Adds 'unit_zon_*' where * = x,y,z
        'unit_fa_*' and 'unit_mer_*' for zonal, field aligned, and meridional
        directions. Note that vector components are expressed in the S/C basis.
        
    """

    # vectors are returned in geo/ecef coordinate system
    add_mag_drift_unit_vectors_ecef(inst, max_steps=max_steps, step_size=step_size)
    # convert them to S/C using transformation supplied by OA
    inst['unit_zon_x'], inst['unit_zon_y'], inst['unit_zon_z'] = project_ecef_vector_onto_basis(inst['unit_zon_ecef_x'], inst['unit_zon_ecef_y'], inst['unit_zon_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_fa_x'], inst['unit_fa_y'], inst['unit_fa_z'] = project_ecef_vector_onto_basis(inst['unit_fa_ecef_x'], inst['unit_fa_ecef_y'], inst['unit_fa_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])
    inst['unit_mer_x'], inst['unit_mer_y'], inst['unit_mer_z'] = project_ecef_vector_onto_basis(inst['unit_mer_ecef_x'], inst['unit_mer_ecef_y'], inst['unit_mer_ecef_z'],
                                                                                                inst['sc_xhat_x'], inst['sc_xhat_y'], inst['sc_xhat_z'],
                                                                                                inst['sc_yhat_x'], inst['sc_yhat_y'], inst['sc_yhat_z'],
                                                                                                inst['sc_zhat_x'], inst['sc_zhat_y'], inst['sc_zhat_z'])

    inst.meta['unit_zon_x'] = { 'long_name':'Zonal direction along IVM-x',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-X component',
                                'axis': 'Zonal Unit Vector: IVM-X component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_y'] = {'long_name':'Zonal direction along IVM-y',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Y component',
                                'axis': 'Zonal Unit Vector: IVM-Y component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_zon_z'] = {'long_name':'Zonal direction along IVM-z',
                                'desc': 'Unit vector for the zonal geomagnetic direction.',
                                'label': 'Zonal Unit Vector: IVM-Z component',
                                'axis': 'Zonal Unit Vector: IVM-Z component',
                                'notes': ('Positive towards the east. Zonal vector is normal to magnetic meridian plane. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_fa_x'] = {'long_name':'Field-aligned direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-X component',
                                'axis': 'Field Aligned Unit Vector: IVM-X component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_y'] = {'long_name':'Field-aligned direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Y component',
                                'axis': 'Field Aligned Unit Vector: IVM-Y component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_fa_z'] = {'long_name':'Field-aligned direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic field line direction.',
                                'label': 'Field Aligned Unit Vector: IVM-Z component',
                                'axis': 'Field Aligned Unit Vector: IVM-Z component',
                                'notes': ('Positive along the field, generally northward. Unit vector is along the geomagnetic field. '
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    inst.meta['unit_mer_x'] = {'long_name':'Meridional direction along IVM-x',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-X component',
                                'axis': 'Meridional Unit Vector: IVM-X component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_y'] = {'long_name':'Meridional direction along IVM-y',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Y component',
                                'axis': 'Meridional Unit Vector: IVM-Y component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}
    inst.meta['unit_mer_z'] = {'long_name':'Meridional direction along IVM-z',
                                'desc': 'Unit vector for the geomagnetic meridional direction.',
                                'label': 'Meridional Unit Vector: IVM-Z component',
                                'axis': 'Meridional Unit Vector: IVM-Z component',
                                'notes': ('Positive is aligned with vertical at '
                                          'geomagnetic equator. Unit vector is perpendicular to the geomagnetic field '
                                          'and in the plane of the meridian.'
                                          'The unit vector is expressed in the IVM coordinate system, x - along RAM, '
                                          'z - towards nadir, y - completes the system, generally southward. '
                                          'Calculated using the corresponding unit vector in ECEF and the orientation '
                                          'of the IVM also expressed in ECEF (sc_*hat_*).'),
                                'scale': 'linear',
                                'units': '',
                               'value_min':-1., 
                               'value_max':1}

    return