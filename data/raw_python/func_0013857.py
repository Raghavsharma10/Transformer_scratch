def add_msis(inst, glat_label='glat', glong_label='glong', 
                                       alt_label='alt'):
    """ 
    Uses MSIS model to obtain thermospheric values.
    
    Uses pyglow module to run MSIS. Configured to use actual solar parameters to run 
    model.
    
    Example
    -------
        # function added velow modifies the inst object upon every inst.load call
        inst.custom.add(add_msis, 'modify', glat_label='custom_label')
    
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
        Input pysat.Instrument object modified to include MSIS values winds.
        'Nn' total neutral density particles/cm^3
        'Nn_N' Nitrogen number density (particles/cm^3)
        'Nn_N2' N2 number density (particles/cm^3)
        'Nn_O' Oxygen number density (particles/cm^3)
        'Nn_O2' O2 number density (particles/cm^3)
        'Tn_msis' Temperature from MSIS (Kelvin)
        
    """
    
    import pyglow
    from pyglow.pyglow import Point
    
    msis_params = []
    # print 'IRI Simulations'
    for time,lat,lon,alt in zip(inst.data.index, inst[glat_label], inst[glong_label], inst[alt_label]):
        pt = Point(time,lat,lon,alt)
        pt.run_msis()
        msis = {}
        total = 0
        for key in pt.nn.keys():
            total += pt.nn[key]
        msis['Nn'] = total
        msis['Nn_N'] = pt.nn['N']
        msis['Nn_N2'] = pt.nn['N2']
        msis['Nn_O'] = pt.nn['O']
        msis['Nn_O2'] = pt.nn['O2']
        msis['Tn_msis'] = pt.Tn_msis
        msis_params.append(msis)        
    # print 'Complete.'
    msis = pds.DataFrame(msis_params)
    msis.index = inst.data.index
    inst[msis.keys()] = msis
    
    # metadata
    inst.meta['Nn'] = {'units':'cm^-3',
                       'desc':'Total neutral number particle density from MSIS.'}
    inst.meta['Nn_N'] = {'units':'cm^-3',
                         'desc':'Total nitrogen number particle density from MSIS.'}
    inst.meta['Nn_N2'] = {'units':'cm^-3',
                          'desc':'Total N2 number particle density from MSIS.'}
    inst.meta['Nn_O'] = {'units':'cm^-3',
                         'desc':'Total oxygen number particle density from MSIS.'}
    inst.meta['Nn_O2'] = {'units':'cm^-3',
                          'desc':'Total O2 number particle density from MSIS.'}
    inst.meta['Tn_msis'] = {'units':'K',
                            'desc':'Neutral temperature from MSIS.'}

    return