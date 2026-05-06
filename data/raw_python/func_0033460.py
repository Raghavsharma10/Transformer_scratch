def apex_location_info(glats, glons, alts, dates):
    """Determine apex location for the field line passing through input point.
    
    Employs a two stage method. A broad step (100 km) field line trace spanning 
    Northern/Southern footpoints is used to find the location with the largest 
    geodetic (WGS84) height. A higher resolution trace (.1 km) is then used to 
    get a better fix on this location. Greatest geodetic height is once again 
    selected.
    
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

    Returns
    -------
    (float, float, float, float, float, float)
        ECEF X (km), ECEF Y (km), ECEF Z (km), 
        Geodetic Latitude (degrees), 
        Geodetic Longitude (degrees), 
        Geodetic Altitude (km)
        
    """

    # use input location and convert to ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)
    # prepare parameters for field line trace
    step_size = 100.
    max_steps = 1000
    steps = np.arange(max_steps)
    # high resolution trace parameters
    fine_step_size = .01
    fine_max_steps = int(step_size/fine_step_size)+10
    fine_steps = np.arange(fine_max_steps)
    # prepare output
    out_x = []
    out_y = []
    out_z = []

    for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                             glats, glons, alts, 
                                                             dates):
        # to get the apex location we need to do a field line trace
        # then find the highest point
        trace = full_field_line(np.array([ecef_x, ecef_y, ecef_z]), date, 0., 
                                steps=steps,
                                step_size=step_size, 
                                max_steps=max_steps)
        # convert all locations to geodetic coordinates
        tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])        
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)
        # repeat using a high resolution trace one big step size each 
        # direction around identified max
        # recurse False ensures only max_steps are taken
        trace = full_field_line(trace[max_idx,:], date, 0., 
                                steps=fine_steps,
                                step_size=fine_step_size, 
                                max_steps=fine_max_steps, 
                                recurse=False)
        # convert all locations to geodetic coordinates
        tlat, tlon, talt = ecef_to_geodetic(trace[:,0], trace[:,1], trace[:,2])
        # determine location that is highest with respect to the geodetic Earth
        max_idx = np.argmax(talt)
        # collect outputs
        out_x.append(trace[max_idx,0])
        out_y.append(trace[max_idx,1])
        out_z.append(trace[max_idx,2])
        
    out_x = np.array(out_x)
    out_y = np.array(out_y)
    out_z = np.array(out_z)
    glat, glon, alt = ecef_to_geodetic(out_x, out_y, out_z)
    
    return out_x, out_y, out_z, glat, glon, alt