def calculate_ecef_velocity(inst):
    """
    Calculates spacecraft velocity in ECEF frame.
    
    Presumes that the spacecraft velocity in ECEF is in 
    the input instrument object as position_ecef_*. Uses a symmetric
    difference to calculate the velocity thus endpoints will be
    set to NaN. Routine should be run using pysat data padding feature
    to create valid end points.
    
    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object
        
    Returns
    -------
    None
        Modifies pysat.Instrument object in place to include ECEF velocity 
        using naming scheme velocity_ecef_* (*=x,y,z)
        
    """
    
    x = inst['position_ecef_x']
    vel_x = (x.values[2:] - x.values[0:-2])/2.

    y = inst['position_ecef_y']
    vel_y = (y.values[2:] - y.values[0:-2])/2.

    z = inst['position_ecef_z']
    vel_z = (z.values[2:] - z.values[0:-2])/2.
    
    inst[1:-1, 'velocity_ecef_x'] = vel_x
    inst[1:-1, 'velocity_ecef_y'] = vel_y
    inst[1:-1, 'velocity_ecef_z'] = vel_z
    
    inst.meta['velocity_ecef_x'] = {'units':'km/s',
                                    'desc':'Velocity of satellite calculated with respect to ECEF frame.'}
    inst.meta['velocity_ecef_y'] = {'units':'km/s',
                                    'desc':'Velocity of satellite calculated with respect to ECEF frame.'}
    inst.meta['velocity_ecef_z'] = {'units':'km/s',
                                    'desc':'Velocity of satellite calculated with respect to ECEF frame.'}
    return