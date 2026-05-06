def step_along_mag_unit_vector(x, y, z, date, direction=None, num_steps=5., 
                               step_size=5., scalar=1):
    """
    Move along 'lines' formed by following the magnetic unit vector directions.

    Moving along the field is effectively the same as a field line trace though
    extended movement along a field should use the specific field_line_trace 
    method.
        
    
    Parameters
    ----------
    x : ECEF-x (km)
        Location to step from in ECEF (km). Scalar input.
    y : ECEF-y (km)
        Location to step from in ECEF (km). Scalar input.
    z : ECEF-z (km)
        Location to step from in ECEF (km). Scalar input.
    date : list-like of datetimes
        Date and time for magnetic field
    direction : string
        String identifier for which unit vector directino to move along.
        Supported inputs, 'meridional', 'zonal', 'aligned'
    num_steps : int
        Number of steps to take along unit vector direction
    step_size = float
        Distance taken for each step (km)
    scalar : int
        Scalar modifier for step size distance. Input a -1 to move along 
        negative unit vector direction.
        
    Returns
    -------
    np.array
        [x, y, z] of ECEF location after taking num_steps along direction, 
        each step_size long.
    
    """
    
    
    # set parameters for the field line tracing routines
    field_step_size = 100.
    field_max_steps = 1000
    field_steps = np.arange(field_max_steps)
    
    for i in np.arange(num_steps):
        # x, y, z in ECEF
        # convert to geodetic
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        # get unit vector directions
        zvx, zvy, zvz, bx, by, bz, mx, my, mz = calculate_mag_drift_unit_vectors_ecef(
                                                        [lat], [lon], [alt], [date],
                                                        steps=field_steps, 
                                                        max_steps=field_max_steps, 
                                                        step_size=field_step_size, 
                                                        ref_height=0.)
        # pull out the direction we need
        if direction == 'meridional':
            ux, uy, uz = mx, my, mz
        elif direction == 'zonal':
            ux, uy, uz = zvx, zvy, zvz
        elif direction == 'aligned':
            ux, uy, uz = bx, by, bz
            
        # take steps along direction
        x = x + step_size*ux[0]*scalar
        y = y + step_size*uy[0]*scalar
        z = z + step_size*uz[0]*scalar
            
    return np.array([x, y, z])