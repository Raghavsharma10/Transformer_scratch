def add_sc_attitude_vectors(inst):
    
    """
    Add attitude vectors for spacecraft assuming ram pointing. 
     
    Presumes spacecraft is pointed along the velocity vector (x), z is
    generally nadir pointing (positive towards Earth), and y completes the 
    right handed system (generally southward).
    
    Notes
    -----
        Expects velocity and position of spacecraft in Earth Centered
        Earth Fixed (ECEF) coordinates to be in the instrument object
        and named velocity_ecef_* (*=x,y,z) and position_ecef_* (*=x,y,z)
    
        Adds attitude vectors for spacecraft in the ECEF basis by calculating the scalar
        product of each attitude vector with each component of ECEF. 

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument object
        
    Returns
    -------
    None
        Modifies pysat.Instrument object in place to include S/C attitude unit
        vectors, expressed in ECEF basis. Vectors are named sc_(x,y,z)hat_ecef_(x,y,z).
        sc_xhat_ecef_x is the spacecraft unit vector along x (positive along velocity vector)
        reported in ECEF, ECEF x-component.

    """
    import pysatMagVect

    # ram pointing is along velocity vector
    inst['sc_xhat_ecef_x'], inst['sc_xhat_ecef_y'], inst['sc_xhat_ecef_z'] = \
        pysatMagVect.normalize_vector(inst['velocity_ecef_x'], inst['velocity_ecef_y'], inst['velocity_ecef_z'])
    
    # begin with z along Nadir (towards Earth)
    # if orbit isn't perfectly circular, then the s/c z vector won't
    # point exactly along nadir. However, nadir pointing is close enough
    # to the true z (in the orbital plane) that we can use it to get y, 
    # and use x and y to get the real z
    inst['sc_zhat_ecef_x'], inst['sc_zhat_ecef_y'], inst['sc_zhat_ecef_z'] = \
        pysatMagVect.normalize_vector(-inst['position_ecef_x'], -inst['position_ecef_y'], -inst['position_ecef_z'])    
    
    # get y vector assuming right hand rule
    # Z x X = Y
    inst['sc_yhat_ecef_x'], inst['sc_yhat_ecef_y'], inst['sc_yhat_ecef_z'] = \
        pysatMagVect.cross_product(inst['sc_zhat_ecef_x'], inst['sc_zhat_ecef_y'], inst['sc_zhat_ecef_z'],
                                   inst['sc_xhat_ecef_x'], inst['sc_xhat_ecef_y'], inst['sc_xhat_ecef_z'])
    # normalize since Xhat and Zhat from above may not be orthogonal
    inst['sc_yhat_ecef_x'], inst['sc_yhat_ecef_y'], inst['sc_yhat_ecef_z'] = \
        pysatMagVect.normalize_vector(inst['sc_yhat_ecef_x'], inst['sc_yhat_ecef_y'], inst['sc_yhat_ecef_z'])
    
    # strictly, need to recalculate Zhat so that it is consistent with RHS
    # just created
    # Z = X x Y      
    inst['sc_zhat_ecef_x'], inst['sc_zhat_ecef_y'], inst['sc_zhat_ecef_z'] = \
        pysatMagVect.cross_product(inst['sc_xhat_ecef_x'], inst['sc_xhat_ecef_y'], inst['sc_xhat_ecef_z'],
                                   inst['sc_yhat_ecef_x'], inst['sc_yhat_ecef_y'], inst['sc_yhat_ecef_z'])
    
    # Adding metadata
    inst.meta['sc_xhat_ecef_x'] = {'units':'', 
                                   'desc':'S/C attitude (x-direction, ram) unit vector, expressed in ECEF basis, x-component'}
    inst.meta['sc_xhat_ecef_y'] = {'units':'',
                                   'desc':'S/C attitude (x-direction, ram) unit vector, expressed in ECEF basis, y-component'}
    inst.meta['sc_xhat_ecef_z'] = {'units':'',
                                   'desc':'S/C attitude (x-direction, ram) unit vector, expressed in ECEF basis, z-component'}
    
    inst.meta['sc_zhat_ecef_x'] = {'units':'',
                                   'desc':'S/C attitude (z-direction, generally nadir) unit vector, expressed in ECEF basis, x-component'}
    inst.meta['sc_zhat_ecef_y'] = {'units':'',
                                   'desc':'S/C attitude (z-direction, generally nadir) unit vector, expressed in ECEF basis, y-component'}
    inst.meta['sc_zhat_ecef_z'] = {'units':'',
                                   'desc':'S/C attitude (z-direction, generally nadir) unit vector, expressed in ECEF basis, z-component'}
    
    inst.meta['sc_yhat_ecef_x'] = {'units':'',
                                   'desc':'S/C attitude (y-direction, generally south) unit vector, expressed in ECEF basis, x-component'}
    inst.meta['sc_yhat_ecef_y'] = {'units':'',
                                   'desc':'S/C attitude (y-direction, generally south) unit vector, expressed in ECEF basis, y-component'}
    inst.meta['sc_yhat_ecef_z'] = {'units':'',
                                   'desc':'S/C attitude (y-direction, generally south) unit vector, expressed in ECEF basis, z-component'}    
    
    # check what magnitudes we get
    mag = np.sqrt(inst['sc_zhat_ecef_x']**2 + inst['sc_zhat_ecef_y']**2 + 
                    inst['sc_zhat_ecef_z']**2)
    idx, = np.where( (mag < .999999999) | (mag > 1.000000001))
    if len(idx) > 0:
        print (mag[idx])
        raise RuntimeError('Unit vector generation failure. Not sufficently orthogonal.')

    return