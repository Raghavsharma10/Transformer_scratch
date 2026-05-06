def closed_loop_edge_lengths_via_footpoint(glats, glons, alts, dates, direction,
                                           vector_direction, step_size=None, 
                                           max_steps=None, edge_length=25., 
                                           edge_steps=5):
    """
    Forms closed loop integration along mag field, satrting at input
    points and goes through footpoint. At footpoint, steps along vector direction
    in both positive and negative directions, then traces back to opposite
    footpoint. Back at input location, steps toward those new field lines 
    (edge_length) along vector direction until hitting distance of minimum
    approach. Loops don't always close. Returns total edge distance 
    that goes through input location, along with the distances of closest approach. 
    
    Note
    ----
        vector direction refers to the magnetic unit vector direction 
    
    Parameters
    ----------
    glats : list-like of floats (degrees)
        Geodetic (WGS84) latitude
    glons : list-like of floats (degrees)
        Geodetic (WGS84) longitude 
    alts : list-like of floats (km)
        Geodetic (WGS84) altitude, height above surface
    dates : list-like of datetimes
        Date and time for determination of scalars
    direction : string
        'north' or 'south' for tracing through northern or
        southern footpoint locations
    vector_direction : string
        'meridional' or 'zonal' unit vector directions
    step_size : float (km)
        Step size (km) used for field line integration
    max_steps : int
        Number of steps taken for field line integration
    edge_length : float (km)
        Half of total edge length (step) taken at footpoint location.
        edge_length step in both positive and negative directions.
    edge_steps : int
        Number of steps taken from footpoint towards new field line
        in a given direction (positive/negative) along unit vector
        
    Returns
    -------
    np.array, np.array, np.array
        A closed loop field line path through input location and footpoint in 
        northern/southern hemisphere and back is taken. The return edge length
        through input location is provided. 
        
        The distances of closest approach for the positive step along vector
        direction, and the negative step are returned.

    
    """
    
    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps)

    if direction == 'south':
        direct = -1
    elif direction == 'north':
        direct = 1

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    full_local_step = []
    min_distance_plus = []
    min_distance_minus = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                             glats, glons, alts, 
                                                             dates):
        # going to try and form close loops via field line integration
        # start at location of interest, map down to northern or southern 
        # footpoints then take symmetric steps along meridional and zonal 
        # directions and trace back from location of interest, step along 
        # field line directions until we intersect or hit the distance of 
        # closest approach to the return field line with the known 
        # distances of footpoint steps, and the closet approach distance
        # we can determine the scalar mapping of one location to another
                    
        yr, doy = pysat.utils.getyrdoy(date)
        double_date = float(yr) + float(doy) / 366.

        # print (glat, glon, alt)
        # trace to footpoint, starting with input location
        sc_root = np.array([ecef_x, ecef_y, ecef_z])
        trace = field_line_trace(sc_root, double_date, direct, 120., 
                                 steps=steps,
                                 step_size=step_size, 
                                 max_steps=max_steps)
        # pull out footpoint location
        ftpnt = trace[-1, :]
        ft_glat, ft_glon, ft_alt = ecef_to_geodetic(*ftpnt)
        
        # take step from footpoint along + vector direction
        plus_step = step_along_mag_unit_vector(ftpnt[0], ftpnt[1], ftpnt[2], 
                                               date, 
                                               direction=vector_direction,
                                               num_steps=edge_steps,
                                               step_size=edge_length/edge_steps)
        # trace this back to other footpoint
        other_plus = field_line_trace(plus_step, double_date, -direct, 0., 
                                      steps=steps,
                                      step_size=step_size, 
                                      max_steps=max_steps)
        # take half step from first footpoint along - vector direction
        minus_step = step_along_mag_unit_vector(ftpnt[0], ftpnt[1], ftpnt[2], 
                                               date, 
                                               direction=vector_direction, 
                                               scalar=-1,
                                               num_steps=edge_steps,
                                               step_size=edge_length/edge_steps)
        # trace this back to other footpoint
        other_minus = field_line_trace(minus_step, double_date, -direct, 0., 
                                       steps=steps,
                                       step_size=step_size, 
                                       max_steps=max_steps)
        # need to determine where the intersection of field line coming back from
        # footpoint through postive vector direction step and back
        # in relation to the vector direction from the s/c location. 
        pos_edge_length, _, mind_pos = step_until_intersect(sc_root,
                                        other_plus,
                                        1, date, 
                                        direction=vector_direction,
                                        field_step_size=step_size,
                                        step_size_goal=edge_length/edge_steps)        
        # take half step from S/C along - vector direction 
        minus_edge_length, _, mind_minus = step_until_intersect(sc_root,
                                        other_minus,
                                        -1, date, 
                                        direction=vector_direction,
                                        field_step_size=step_size,
                                        step_size_goal=edge_length/edge_steps)
        # collect outputs
        full_local_step.append(pos_edge_length + minus_edge_length)
        min_distance_plus.append(mind_pos)
        min_distance_minus.append(mind_minus)
        
    return np.array(full_local_step), np.array(min_distance_plus), np.array(min_distance_minus)