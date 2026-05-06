def add_hwm_winds_and_ecef_vectors(inst, glat_label='glat', glong_label='glong', 
                                         alt_label='alt'):
    """ 
    Uses HWM (Horizontal Wind Model) model to obtain neutral wind details.
    
    Uses pyglow module to run HWM. Configured to use actual solar parameters to run 
    model.
    
    Example
    -------
        # function added velow modifies the inst object upon every inst.load call
        inst.custom.add(add_hwm_winds_and_ecef_vectors, 'modify', glat_label='custom_label')
    
    Parameters
    ----------
    inst : pysat.Instrument
        Designed with pysat_sgp4 in mind
    glat_label : string
        label used in inst to identify WGS84 geodetic latitude (degrees)
    glong_label : string
        label used in inst to identify WGS84 geodetic longitude (degrees)
    alt_label : string
        label used in inst to identify WGS84 geodetic altitude (km, height above surface)
        
    Returns
    -------
    inst
        Input pysat.Instrument object modified to include HWM winds.
        'zonal_wind' for the east/west winds (u in model) in m/s
        'meiridional_wind' for the north/south winds (v in model) in m/s
        'unit_zonal_wind_ecef_*' (*=x,y,z) is the zonal vector expressed in the ECEF basis
        'unit_mer_wind_ecef_*' (*=x,y,z) is the meridional vector expressed in the ECEF basis
        'sim_inst_wind_*' (*=x,y,z) is the projection of the total wind vector onto s/c basis
        
    """

    import pyglow
    import pysatMagVect

    hwm_params = []
    for time,lat,lon,alt in zip(inst.data.index, inst[glat_label], inst[glong_label], inst[alt_label]):
        # Point class is instantiated. 
        # Its parameters are a function of time and spatial location
        pt = pyglow.Point(time,lat,lon,alt)
        pt.run_hwm()
        hwm = {}
        hwm['zonal_wind'] = pt.u
        hwm['meridional_wind'] = pt.v
        hwm_params.append(hwm)        
    # print 'Complete.'
    hwm = pds.DataFrame(hwm_params)
    hwm.index = inst.data.index
    inst[['zonal_wind', 'meridional_wind']] = hwm[['zonal_wind', 'meridional_wind']]
    
    # calculate zonal unit vector in ECEF
    # zonal wind: east - west; positive east
    # EW direction is tangent to XY location of S/C in ECEF coordinates
    mag = np.sqrt(inst['position_ecef_x']**2 + inst['position_ecef_y']**2)
    inst['unit_zonal_wind_ecef_x'] = -inst['position_ecef_y']/mag
    inst['unit_zonal_wind_ecef_y'] = inst['position_ecef_x']/mag
    inst['unit_zonal_wind_ecef_z'] = 0*inst['position_ecef_x']
    
    # calculate meridional unit vector in ECEF
    # meridional wind: north - south; positive north
    # mer direction completes RHS of position and zonal vector
    unit_pos_x, unit_pos_y, unit_pos_z = \
        pysatMagVect.normalize_vector(-inst['position_ecef_x'], -inst['position_ecef_y'], -inst['position_ecef_z'])    
    
    # mer = r x zonal
    inst['unit_mer_wind_ecef_x'], inst['unit_mer_wind_ecef_y'], inst['unit_mer_wind_ecef_z'] = \
        pysatMagVect.cross_product(unit_pos_x, unit_pos_y, unit_pos_z,
                                   inst['unit_zonal_wind_ecef_x'], inst['unit_zonal_wind_ecef_y'], inst['unit_zonal_wind_ecef_z'])
    
    # Adding metadata information                                
    inst.meta['zonal_wind'] = {'units':'m/s','long_name':'Zonal Wind', 
                               'desc':'HWM model zonal wind'}
    inst.meta['meridional_wind'] = {'units':'m/s','long_name':'Meridional Wind', 
                                    'desc':'HWM model meridional wind'}
    inst.meta['unit_zonal_wind_ecef_x'] = {'units':'',
                                           'long_name':'Zonal Wind Unit ECEF x-vector', 
                                           'desc':'x-value of zonal wind unit vector in ECEF co ordinates'}
    inst.meta['unit_zonal_wind_ecef_y'] = {'units':'', 
                                           'long_name':'Zonal Wind Unit ECEF y-vector', 
                                           'desc':'y-value of zonal wind unit vector in ECEF co ordinates'}
    inst.meta['unit_zonal_wind_ecef_z'] = {'units':'',
                                           'long_name':'Zonal Wind Unit ECEF z-vector', 
                                           'desc':'z-value of zonal wind unit vector in ECEF co ordinates'}
    inst.meta['unit_mer_wind_ecef_x'] = {'units':'',
                                         'long_name':'Meridional Wind Unit ECEF x-vector', 
                                         'desc':'x-value of meridional wind unit vector in ECEF co ordinates'}
    inst.meta['unit_mer_wind_ecef_y'] = {'units':'',
                                         'long_name':'Meridional Wind Unit ECEF y-vector', 
                                         'desc':'y-value of meridional wind unit vector in ECEF co ordinates'}
    inst.meta['unit_mer_wind_ecef_z'] = {'units':'',
                                         'long_name':'Meridional Wind Unit ECEF z-vector', 
                                         'desc':'z-value of meridional wind unit vector in ECEF co ordinates'}
    return