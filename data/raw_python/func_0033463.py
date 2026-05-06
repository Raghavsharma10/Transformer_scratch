def scalars_for_mapping_ion_drifts(glats, glons, alts, dates, step_size=None, 
                                   max_steps=None, e_field_scaling_only=False):
    """
    Calculates scalars for translating ion motions at position
    glat, glon, and alt, for date, to the footpoints of the field line
    as well as at the magnetic equator.
    
    All inputs are assumed to be 1D arrays.
    
    Note
    ----
        Directions refer to the ion motion direction e.g. the zonal
        scalar applies to zonal ion motions (meridional E field assuming ExB ion motion)
    
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
    e_field_scaling_only : boolean (False)
        If True, method only calculates the electric field scalar, ignoring 
        changes in magnitude of B. Note ion velocity related to E/B.
        
    Returns
    -------
    dict
        array-like of scalars for translating ion drifts. Keys are,
        'north_zonal_drifts_scalar', 'north_mer_drifts_scalar', and similarly
        for southern locations. 'equator_mer_drifts_scalar' and 
        'equator_zonal_drifts_scalar' cover the mappings to the equator.
    
    """
    
    if step_size is None:
        step_size = 100.
    if max_steps is None:
        max_steps = 1000
    steps = np.arange(max_steps)

    # use spacecraft location to get ECEF
    ecef_xs, ecef_ys, ecef_zs = geodetic_to_ecef(glats, glons, alts)

    # prepare output
    eq_zon_drifts_scalar = []
    eq_mer_drifts_scalar = []
    # magnetic field info
    north_mag_scalar = []
    south_mag_scalar = []
    eq_mag_scalar = []
    out = {}
    # meridional e-field scalar map, can also be
    # zonal ion drift scalar map
    # print ('Starting Northern')
    north_zon_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'north',
                                                        'meridional',
                                                        step_size=step_size,
                                                        max_steps=max_steps, 
                                                        edge_length=25.,
                                                        edge_steps=5)

    north_mer_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'north',
                                                        'zonal',
                                                        step_size=step_size,
                                                        max_steps=max_steps, 
                                                        edge_length=25.,
                                                        edge_steps=5)

    # print ('Starting Southern')
    south_zon_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'south',
                                                        'meridional',
                                                        step_size=step_size,
                                                        max_steps=max_steps, 
                                                        edge_length=25.,
                                                        edge_steps=5)

    south_mer_drifts_scalar, mind_plus, mind_minus = closed_loop_edge_lengths_via_footpoint(glats,
                                                        glons, alts, dates, 'south',
                                                        'zonal',
                                                        step_size=step_size,
                                                        max_steps=max_steps, 
                                                        edge_length=25.,
                                                        edge_steps=5)
    # print ('Starting Equatorial')                
    # , step_zon_apex2, mind_plus, mind_minus                                 
    eq_zon_drifts_scalar = closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                                        'meridional',
                                                        edge_length=25., 
                                                        edge_steps=5)
    # , step_mer_apex2, mind_plus, mind_minus                                                      
    eq_mer_drifts_scalar = closed_loop_edge_lengths_via_equator(glats, glons, alts, dates,
                                                        'zonal',
                                                        edge_length=25., 
                                                        edge_steps=5)
    # print ('Done with core')
    north_zon_drifts_scalar = north_zon_drifts_scalar/50. 
    south_zon_drifts_scalar = south_zon_drifts_scalar/50. 
    north_mer_drifts_scalar = north_mer_drifts_scalar/50. 
    south_mer_drifts_scalar = south_mer_drifts_scalar/50. 
    # equatorial 
    eq_zon_drifts_scalar = 50./eq_zon_drifts_scalar
    eq_mer_drifts_scalar = 50./eq_mer_drifts_scalar

    if e_field_scaling_only:
        # prepare output
        out['north_mer_fields_scalar'] = north_zon_drifts_scalar
        out['south_mer_fields_scalar'] = south_zon_drifts_scalar
        out['north_zon_fields_scalar'] = north_mer_drifts_scalar
        out['south_zon_fields_scalar'] = south_mer_drifts_scalar
        out['equator_mer_fields_scalar'] = eq_zon_drifts_scalar
        out['equator_zon_fields_scalar'] = eq_mer_drifts_scalar
    
    else:
        # figure out scaling for drifts based upon change in magnetic field
        # strength
        for ecef_x, ecef_y, ecef_z, glat, glon, alt, date in zip(ecef_xs, ecef_ys, ecef_zs, 
                                                                glats, glons, alts, 
                                                                dates):            
            yr, doy = pysat.utils.getyrdoy(date)
            double_date = float(yr) + float(doy) / 366.
            # get location of apex for s/c field line
            apex_x, apex_y, apex_z, apex_lat, apex_lon, apex_alt = apex_location_info(
                                                                        [glat], [glon], 
                                                                        [alt], [date])    
            # trace to northern footpoint
            sc_root = np.array([ecef_x, ecef_y, ecef_z])
            trace_north = field_line_trace(sc_root, double_date, 1., 120., 
                                        steps=steps,
                                        step_size=step_size, 
                                        max_steps=max_steps)
            # southern tracing
            trace_south = field_line_trace(sc_root, double_date, -1., 120., 
                                        steps=steps,
                                        step_size=step_size, 
                                        max_steps=max_steps)
            # footpoint location
            north_ftpnt = trace_north[-1, :]
            nft_glat, nft_glon, nft_alt = ecef_to_geodetic(*north_ftpnt)
            south_ftpnt = trace_south[-1, :]
            sft_glat, sft_glon, sft_alt = ecef_to_geodetic(*south_ftpnt)
    
            # scalar for the northern footpoint electric field based on distances
            # for drift also need to include the magnetic field, drift = E/B
            tbn, tbe, tbd, b_sc = igrf.igrf12syn(0, double_date, 1, alt, 
                                                np.deg2rad(90.-glat), 
                                                np.deg2rad(glon))
            # get mag field and scalar for northern footpoint
            tbn, tbe, tbd, b_nft = igrf.igrf12syn(0, double_date, 1, nft_alt, 
                                                np.deg2rad(90.-nft_glat), 
                                                np.deg2rad(nft_glon))
            north_mag_scalar.append(b_sc/b_nft)            
            # equatorial values
            tbn, tbe, tbd, b_eq = igrf.igrf12syn(0, double_date, 1, apex_alt, 
                                                 np.deg2rad(90.-apex_lat), 
                                                 np.deg2rad(apex_lon))
            eq_mag_scalar.append(b_sc/b_eq)        
            # scalar for the southern footpoint
            tbn, tbe, tbd, b_sft = igrf.igrf12syn(0, double_date, 1, sft_alt, 
                                                  np.deg2rad(90.-sft_glat), 
                                                  np.deg2rad(sft_glon))
            south_mag_scalar.append(b_sc/b_sft)
        
        # make E-Field scalars to drifts
        # lists to arrays
        north_mag_scalar = np.array(north_mag_scalar)
        south_mag_scalar = np.array(south_mag_scalar)
        eq_mag_scalar = np.array(eq_mag_scalar)
        # apply to electric field scaling to get ion drift values
        north_zon_drifts_scalar = north_zon_drifts_scalar*north_mag_scalar
        south_zon_drifts_scalar = south_zon_drifts_scalar*south_mag_scalar 
        north_mer_drifts_scalar = north_mer_drifts_scalar*north_mag_scalar
        south_mer_drifts_scalar = south_mer_drifts_scalar*south_mag_scalar
        # equatorial 
        eq_zon_drifts_scalar = eq_zon_drifts_scalar*eq_mag_scalar
        eq_mer_drifts_scalar = eq_mer_drifts_scalar*eq_mag_scalar
        # output
        out['north_zonal_drifts_scalar'] = north_zon_drifts_scalar
        out['south_zonal_drifts_scalar'] = south_zon_drifts_scalar
        out['north_mer_drifts_scalar'] = north_mer_drifts_scalar
        out['south_mer_drifts_scalar'] = south_mer_drifts_scalar
        out['equator_zonal_drifts_scalar'] = eq_zon_drifts_scalar
        out['equator_mer_drifts_scalar'] = eq_mer_drifts_scalar

    return out