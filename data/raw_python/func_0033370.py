def add_mag_drift_unit_vectors_ecef(inst, steps=None, max_steps=40000, step_size=10.,
                                    ref_height=120.):
    """Adds unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object that will get unit vectors
    max_steps : int
        Maximum number of steps allowed for field line tracing
    step_size : float
        Maximum step size (km) allowed when field line tracing
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
        
    Returns
    -------
    None
        unit vectors are added to the passed Instrument object with a naming 
        scheme:
            'unit_zon_ecef_*' : unit zonal vector, component along ECEF-(X,Y,or Z)
            'unit_fa_ecef_*' : unit field-aligned vector, component along ECEF-(X,Y,or Z)
            'unit_mer_ecef_*' : unit meridional vector, component along ECEF-(X,Y,or Z)
            
    """

    # add unit vectors for magnetic drifts in ecef coordinates
    zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(inst['latitude'], 
                                                            inst['longitude'], inst['altitude'], inst.data.index,
                                                            steps=steps, max_steps=max_steps, step_size=step_size, ref_height=ref_height)
    
    inst['unit_zon_ecef_x'] = zvx
    inst['unit_zon_ecef_y'] = zvy
    inst['unit_zon_ecef_z'] = zvz

    inst['unit_fa_ecef_x'] = bx
    inst['unit_fa_ecef_y'] = by
    inst['unit_fa_ecef_z'] = bz

    inst['unit_mer_ecef_x'] = mx
    inst['unit_mer_ecef_y'] = my
    inst['unit_mer_ecef_z'] = mz

    inst.meta['unit_zon_ecef_x'] = {'long_name': 'Zonal unit vector along ECEF-x',
                                    'desc': 'Zonal unit vector along ECEF-x',
                                    'label': 'Zonal unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_y'] = {'long_name': 'Zonal unit vector along ECEF-y',
                                    'desc': 'Zonal unit vector along ECEF-y',
                                    'label': 'Zonal unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_zon_ecef_z'] = {'long_name': 'Zonal unit vector along ECEF-z',
                                    'desc': 'Zonal unit vector along ECEF-z',
                                    'label': 'Zonal unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Zonal unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_fa_ecef_x'] = {'long_name': 'Field-aligned unit vector along ECEF-x',
                                    'desc': 'Field-aligned unit vector along ECEF-x',
                                    'label': 'Field-aligned unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_y'] = {'long_name': 'Field-aligned unit vector along ECEF-y',
                                    'desc': 'Field-aligned unit vector along ECEF-y',
                                    'label': 'Field-aligned unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_fa_ecef_z'] = {'long_name': 'Field-aligned unit vector along ECEF-z',
                                    'desc': 'Field-aligned unit vector along ECEF-z',
                                    'label': 'Field-aligned unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Field-aligned unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    inst.meta['unit_mer_ecef_x'] = {'long_name': 'Meridional unit vector along ECEF-x',
                                    'desc': 'Meridional unit vector along ECEF-x',
                                    'label': 'Meridional unit vector along ECEF-x',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-x',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_y'] = {'long_name': 'Meridional unit vector along ECEF-y',
                                    'desc': 'Meridional unit vector along ECEF-y',
                                    'label': 'Meridional unit vector along ECEF-y',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-y',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }
    inst.meta['unit_mer_ecef_z'] = {'long_name': 'Meridional unit vector along ECEF-z',
                                    'desc': 'Meridional unit vector along ECEF-z',
                                    'label': 'Meridional unit vector along ECEF-z',
                                    'notes': ('Unit vector expressed using Earth Centered Earth Fixed (ECEF) frame. '
                                              'Vector system is calcluated by field-line tracing along IGRF values '
                                              'down to reference altitudes of 120 km in both the Northern and Southern '
                                              'hemispheres. These two points, along with the satellite position, are '
                                              'used to define the magnetic meridian. Vector math from here generates '
                                              'the orthogonal system.'),
                                    'axis': 'Meridional unit vector along ECEF-z',
                                    'value_min': -1.,
                                    'value_max': 1.,
                                    }

    return