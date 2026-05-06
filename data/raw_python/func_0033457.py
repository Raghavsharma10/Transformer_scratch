def calculate_mag_drift_unit_vectors_ecef(latitude, longitude, altitude, datetimes,
                                          steps=None, max_steps=1000, step_size=100.,
                                          ref_height=120., filter_zonal=True):
    """Calculates unit vectors expressing the ion drift coordinate system
    organized by the geomagnetic field. Unit vectors are expressed
    in ECEF coordinates.
    
    Note
    ----
        The zonal vector is calculated by field-line tracing from
        the input locations toward the footpoint locations at ref_height.
        The cross product of these two vectors is taken to define the plane of
        the magnetic field. This vector is not always orthogonal
        with the local field-aligned vector (IGRF), thus any component of the 
        zonal vector with the field-aligned direction is removed (optional). 
        The meridional unit vector is defined via the cross product of the 
        zonal and field-aligned directions.
    
    Parameters
    ----------
    latitude : array-like of floats (degrees)
        Latitude of location, degrees, WGS84
    longitude : array-like of floats (degrees)
        Longitude of location, degrees, WGS84
    altitude : array-like of floats (km)
        Altitude of location, height above surface, WGS84
    datetimes : array-like of datetimes
        Time to calculate vectors
    max_steps : int
        Maximum number of steps allowed for field line tracing
    step_size : float
        Maximum step size (km) allowed when field line tracing
    ref_height : float
        Altitude used as cutoff for labeling a field line location a footpoint
    filter_zonal : bool
        If True, removes any field aligned component from the calculated
        zonal unit vector. Resulting coordinate system is not-orthogonal.
        
    Returns
    -------
    zon_x, zon_y, zon_z, fa_x, fa_y, fa_z, mer_x, mer_y, mer_z
            
    """

    if steps is None:
        steps = np.arange(max_steps)
    # calculate satellite position in ECEF coordinates
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(latitude, longitude, altitude)
    # also get position in geocentric coordinates
    geo_lat, geo_long, geo_alt = ecef_to_geocentric(ecef_x, ecef_y, ecef_z, 
                                                    ref_height=0.)
    # filter longitudes (could use pysat's function here)
    idx, = np.where(geo_long < 0)
    geo_long[idx] = geo_long[idx] + 360.
    # prepare output lists
    north_x = [];
    north_y = [];
    north_z = []
    south_x = [];
    south_y = [];
    south_z = []
    bn = [];
    be = [];
    bd = []

    for x, y, z, alt, colat, elong, time in zip(ecef_x, ecef_y, ecef_z, 
                                                geo_alt, np.deg2rad(90. - geo_lat),
                                                np.deg2rad(geo_long), datetimes):
        init = np.array([x, y, z])
        # date = inst.yr + inst.doy / 366.
        # trace = full_field_line(init, time, ref_height, step_size=step_size, 
        #                                                 max_steps=max_steps,
        #                                                 steps=steps)
        trace_north = field_line_trace(init, time, 1., ref_height, steps=steps,
                                        step_size=step_size, max_steps=max_steps)
        trace_south = field_line_trace(init, time, -1., ref_height, steps=steps,
                                        step_size=step_size, max_steps=max_steps)
        # store final location, full trace goes south to north
        trace_north = trace_north[-1, :]
        trace_south = trace_south[-1, :]
        # magnetic field at spacecraft location, using geocentric inputs
        # to get magnetic field in geocentric output
        # recast from datetime to float, as required by IGRF12 code
        doy = (time - datetime.datetime(time.year,1,1)).days
        # number of days in year, works for leap years
        num_doy_year = (datetime.datetime(time.year+1,1,1) - datetime.datetime(time.year,1,1)).days
        date = time.year + float(doy)/float(num_doy_year) + (time.hour + time.minute/60. + time.second/3600.)/24.
        # get IGRF field components
        # tbn, tbe, tbd, tbmag are in nT
        tbn, tbe, tbd, tbmag = igrf.igrf12syn(0, date, 1, alt, colat, elong)
        # collect outputs
        south_x.append(trace_south[0])
        south_y.append(trace_south[1])
        south_z.append(trace_south[2])
        north_x.append(trace_north[0])
        north_y.append(trace_north[1])
        north_z.append(trace_north[2])

        bn.append(tbn);
        be.append(tbe);
        bd.append(tbd)

    north_x = np.array(north_x)
    north_y = np.array(north_y)
    north_z = np.array(north_z)
    south_x = np.array(south_x)
    south_y = np.array(south_y)
    south_z = np.array(south_z)
    bn = np.array(bn)
    be = np.array(be)
    bd = np.array(bd)

    # calculate vector from satellite to northern/southern footpoints
    north_x = north_x - ecef_x
    north_y = north_y - ecef_y
    north_z = north_z - ecef_z
    north_x, north_y, north_z = normalize_vector(north_x, north_y, north_z)
    south_x = south_x - ecef_x
    south_y = south_y - ecef_y
    south_z = south_z - ecef_z
    south_x, south_y, south_z = normalize_vector(south_x, south_y, south_z)
    # calculate magnetic unit vector
    bx, by, bz = enu_to_ecef_vector(be, bn, -bd, geo_lat, geo_long)
    bx, by, bz = normalize_vector(bx, by, bz)
    
    # take cross product of southward and northward vectors to get the zonal vector
    zvx_foot, zvy_foot, zvz_foot = cross_product(south_x, south_y, south_z,
                                                 north_x, north_y, north_z)  
    # getting zonal vector utilizing magnetic field vector instead
    zvx_north, zvy_north, zvz_north = cross_product(north_x, north_y, north_z,
                                                    bx, by, bz)
    # getting zonal vector utilizing magnetic field vector instead and southern point
    zvx_south, zvy_south, zvz_south = cross_product(south_x, south_y, south_z,
                                                    bx, by, bz)
    # normalize the vectors
    norm_foot = np.sqrt(zvx_foot ** 2 + zvy_foot ** 2 + zvz_foot ** 2)
    
    # calculate zonal vector
    zvx = zvx_foot / norm_foot
    zvy = zvy_foot / norm_foot
    zvz = zvz_foot / norm_foot
    # remove any field aligned component to the zonal vector
    dot_fa = zvx * bx + zvy * by + zvz * bz
    zvx -= dot_fa * bx
    zvy -= dot_fa * by
    zvz -= dot_fa * bz
    zvx, zvy, zvz = normalize_vector(zvx, zvy, zvz)
    # compute meridional vector
    # cross product of zonal and magnetic unit vector
    mx, my, mz = cross_product(zvx, zvy, zvz,
                               bx, by, bz)
    # add unit vectors for magnetic drifts in ecef coordinates
    return zvx, zvy, zvz, bx, by, bz, mx, my, mz