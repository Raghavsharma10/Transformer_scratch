def closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                         vector_direction,
                                         edge_length=25., 
                                         edge_steps=5):
    """
    Calculates the distance between apex locations mapping to the input location.
    
    Using the input location, the apex location is calculated. Also from the input 
    location, a step along both the positive and negative
    vector_directions is taken, and the apex locations for those points are calculated.
    The difference in position between these apex locations is the total centered
    distance between magnetic field lines at the magnetic apex when starting
    locally with a field line half distance of edge_length.
    
    An alternative method has been implemented, then commented out.
    This technique takes multiple steps from the origin apex towards the apex
    locations identified along vector_direction. In principle this is more accurate
    but more computationally intensive, similar to the footpoint model.
    A comparison is planned.
    
    
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
    np.array, ### np.array, np.array
        The change in field line apex locations. 
        
        ## Pending ## The return edge length through input location is provided. 
        
        ## Pending ## The distances of closest approach for the positive step 
                      along vector direction, and the negative step are returned.

    
    """

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    apex_edge_length = []
    # outputs for alternative calculation
    full_local_step = []
    min_distance_plus = []
    min_distance_minus = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                             glats, glons, alts, 
                                                             dates):
        
        yr, doy = pysat.utils.getyrdoy(date)
        double_date = float(yr) + float(doy) / 366.
                    
        # get location of apex for s/c field line
        apex_x, apex_y, apex_z, apex_lat, apex_lon, apex_alt = apex_location_info(
                                                                    [glat], [glon], 
                                                                    [alt], [date])
        # apex in ecef (maps to input location)
        apex_root = np.array([apex_x[0], apex_y[0], apex_z[0]])      
        # take step from s/c along + vector direction
        # then get the apex location
        plus = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, 
                                          direction=vector_direction,
                                          num_steps=edge_steps,
                                          step_size=edge_length/edge_steps)
        plus_lat, plus_lon, plus_alt = ecef_to_geodetic(*plus)
        plus_apex_x, plus_apex_y, plus_apex_z, plus_apex_lat, plus_apex_lon, plus_apex_alt = \
                    apex_location_info([plus_lat], [plus_lon], [plus_alt], [date])
        # plus apex location in ECEF
        plus_apex_root = np.array([plus_apex_x[0], plus_apex_y[0], plus_apex_z[0]])   

        # take half step from s/c along - vector direction
        # then get the apex location
        minus = step_along_mag_unit_vector(ecef_x, ecef_y, ecef_z, date, 
                                               direction=vector_direction, 
                                               scalar=-1,
                                               num_steps=edge_steps,
                                               step_size=edge_length/edge_steps)
        minus_lat, minus_lon, minus_alt = ecef_to_geodetic(*minus)
        minus_apex_x, minus_apex_y, minus_apex_z, minus_apex_lat, minus_apex_lon, minus_apex_alt = \
                    apex_location_info([minus_lat], [minus_lon], [minus_alt], [date])
        minus_apex_root = np.array([minus_apex_x[0], minus_apex_y[0], minus_apex_z[0]])   

        # take difference in apex locations
        apex_edge_length.append(np.sqrt((plus_apex_x[0]-minus_apex_x[0])**2 + 
                                        (plus_apex_y[0]-minus_apex_y[0])**2 + 
                                        (plus_apex_z[0]-minus_apex_z[0])**2))

#         # take an alternative path to calculation
#         # do field line trace around pos and neg apexes
#         # then do intersection with field line projection thing        
# 
#         # do a short centered field line trace around plus apex location
#         other_trace = full_field_line(plus_apex_root, double_date, 0., 
#                                       step_size=1., 
#                                       max_steps=10,
#                                       recurse=False)
#         # need to determine where the intersection of apex field line 
#         # in relation to the vector direction from the s/c field apex location.
#         pos_edge_length, _, mind_pos = step_until_intersect(apex_root,
                                        # other_trace,
                                        # 1, date, 
                                        # direction=vector_direction,
                                        # field_step_size=1.,
                                        # step_size_goal=edge_length/edge_steps)                                                                                               
#         # do a short centered field line trace around 'minus' apex location
#         other_trace = full_field_line(minus_apex_root, double_date, 0., 
#                                       step_size=1., 
#                                       max_steps=10,
#                                       recurse=False)
#         # need to determine where the intersection of apex field line 
#         # in relation to the vector direction from the s/c field apex location. 
#         minus_edge_length, _, mind_minus = step_until_intersect(apex_root,
                                        # other_trace,
                                        # -1, date, 
                                        # direction=vector_direction,
                                        # field_step_size=1.,
                                        # step_size_goal=edge_length/edge_steps)        
        # full_local_step.append(pos_edge_length + minus_edge_length)
        # min_distance_plus.append(mind_pos)
        # min_distance_minus.append(mind_minus)
        
        # still sorting out alternative option for this calculation
        # commented code is 'good' as far as the plan goes
        # takes more time, so I haven't tested one vs the other yet
        # having two live methods can lead to problems
        # THIS IS A TODO (sort it out)
    return np.array(apex_edge_length)