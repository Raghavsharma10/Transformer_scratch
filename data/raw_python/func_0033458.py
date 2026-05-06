def step_until_intersect(pos, field_line, sign, time,  direction=None,
                        step_size_goal=5., 
                        field_step_size=None):   
    """Starting at pos, method steps along magnetic unit vector direction 
    towards the supplied field line trace. Determines the distance of 
    closest approach to field line.
    
    Routine is used when calculting the mapping of electric fields along 
    magnetic field lines. Voltage remains constant along the field but the 
    distance between field lines does not.This routine may be used to form the 
    last leg when trying to trace out a closed field line loop.
    
    Routine will create a high resolution field line trace (.01 km step size) 
    near the location of closest approach to better determine where the 
    intersection occurs. 
    
    Parameters
    ----------
    pos : array-like
        X, Y, and Z ECEF locations to start from
    field_line : array-like (:,3)
        X, Y, and Z ECEF locations of field line trace, produced by the
        field_line_trace method.
    sign : int
        if 1, move along positive unit vector. Negwtive direction for -1.
    time : datetime or float
        Date to perform tracing on (year + day/365 + hours/24. + etc.)
        Accounts for leap year if datetime provided.
    direction : string ('meridional', 'zonal', or 'aligned')
        Which unit vector direction to move slong when trying to intersect
        with supplied field line trace. See step_along_mag_unit_vector method
        for more.
    step_size_goal : float
        step size goal that method will try to match when stepping towards field line. 
    
    Returns
    -------
    (float, array, float)
        Total distance taken along vector direction; the position after taking 
        the step [x, y, z] in ECEF; distance of closest approach from input pos 
        towards the input field line trace.
         
    """ 
                                                         
    # work on a copy, probably not needed
    field_copy = field_line
    # set a high last minimum distance to ensure first loop does better than this
    last_min_dist = 2500000.
    # scalar is the distance along unit vector line that we are taking
    scalar = 0.
    # repeat boolean
    repeat=True
    # first run boolean
    first=True
    # factor is a divisor applied to the remaining distance between point and field line
    # I slowly take steps towards the field line and I don't want to overshoot
    # each time my minimum distance increases, I step back, increase factor, reducing
    # my next step size, then I try again
    factor = 1
    while repeat:
        # take a total step along magnetic unit vector
        # try to take steps near user provided step_size_goal
        unit_steps = np.abs(scalar//step_size_goal)
        if unit_steps == 0:
            unit_steps = 1
        # print (unit_steps, scalar/unit_steps)
        pos_step = step_along_mag_unit_vector(pos[0], pos[1], pos[2], time, 
                                              direction=direction,
                                              num_steps=unit_steps, 
                                              step_size=np.abs(scalar)/unit_steps,
                                              scalar=sign) 
        # find closest point along field line trace
        diff = field_copy - pos_step
        diff_mag = np.sqrt((diff ** 2).sum(axis=1))
        min_idx = np.argmin(diff_mag)
        if first:
            # first time in while loop, create some information
            # make a high resolution field line trace around closest distance
            # want to take a field step size in each direction
            # maintain accuracy of high res trace below to be .01 km
            init = field_copy[min_idx,:]
            field_copy = full_field_line(init, time, 0.,
                                         step_size=0.01, 
                                         max_steps=int(field_step_size/.01),
                                         recurse=False)
            # difference with position
            diff = field_copy - pos_step
            diff_mag = np.sqrt((diff ** 2).sum(axis=1))
            # find closest one
            min_idx = np.argmin(diff_mag)
            # # reduce number of elements we really need to check
            # field_copy = field_copy[min_idx-100:min_idx+100]
            # # difference with position
            # diff = field_copy - pos_step
            # diff_mag = np.sqrt((diff ** 2).sum(axis=1))
            # # find closest one
            # min_idx = np.argmin(diff_mag)
            first = False
            
        # pull out distance of closest point 
        min_dist = diff_mag[min_idx]
        
        # check how the solution is doing
        # if well, add more distance to the total step and recheck if closer
        # if worse, step back and try a smaller step
        if min_dist > last_min_dist:
            # last step we took made the solution worse
            if factor > 4:
                # we've tried enough, stop looping
                repeat = False
                # undo increment to last total distance
                scalar = scalar - last_min_dist/(2*factor)
                # calculate latest position
                pos_step = step_along_mag_unit_vector(pos[0], pos[1], pos[2], 
                                        time, 
                                        direction=direction,
                                        num_steps=unit_steps, 
                                        step_size=np.abs(scalar)/unit_steps,
                                        scalar=sign) 
            else:
                # undo increment to last total distance
                scalar = scalar - last_min_dist/(2*factor)
                # increase the divisor used to reduce the distance 
                # actually stepped per increment
                factor = factor + 1.
                # try a new increment to total distance
                scalar = scalar + last_min_dist/(2*factor)
        else:
            # we did better, move even closer, a fraction of remaining distance
            # increment scalar, but only by a fraction
            scalar = scalar + min_dist/(2*factor)
            # we have a new standard to judge against, set it
            last_min_dist = min_dist.copy()

    # return magnitude of step
    return scalar, pos_step, min_dist