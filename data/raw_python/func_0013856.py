def add_igrf(inst, glat_label='glat', glong_label='glong', 
                                       alt_label='alt'):
    """ 
    Uses International Geomagnetic Reference Field (IGRF) model to obtain geomagnetic field values.
    
    Uses pyglow module to run IGRF. Configured to use actual solar parameters to run 
    model.
    
    Example
    -------
        # function added velow modifies the inst object upon every inst.load call
        inst.custom.add(add_igrf, 'modify', glat_label='custom_label')
    
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
        'B' total geomagnetic field
        'B_east' Geomagnetic field component along east/west directions (+ east)
        'B_north' Geomagnetic field component along north/south directions (+ north)
        'B_up' Geomagnetic field component along up/down directions (+ up)
        'B_ecef_x' Geomagnetic field component along ECEF x
        'B_ecef_y' Geomagnetic field component along ECEF y
        'B_ecef_z' Geomagnetic field component along ECEF z
        
    """
    
    import pyglow
    from pyglow.pyglow import Point
    import pysatMagVect
    
    igrf_params = []
    # print 'IRI Simulations'
    for time,lat,lon,alt in zip(inst.data.index, inst[glat_label], inst[glong_label], inst[alt_label]):
        pt = Point(time,lat,lon,alt)
        pt.run_igrf()
        igrf = {}
        igrf['B'] = pt.B
        igrf['B_east'] = pt.Bx
        igrf['B_north'] = pt.By
        igrf['B_up'] = pt.Bz
        igrf_params.append(igrf)        
    # print 'Complete.'
    igrf = pds.DataFrame(igrf_params)
    igrf.index = inst.data.index
    inst[igrf.keys()] = igrf
    
    # convert magnetic field in East/north/up to ECEF basis
    x, y, z = pysatMagVect.enu_to_ecef_vector(inst['B_east'], inst['B_north'], inst['B_up'],
                                              inst[glat_label], inst[glong_label])
    inst['B_ecef_x'] = x
    inst['B_ecef_y'] = y
    inst['B_ecef_z'] = z
    
    # metadata
    inst.meta['B'] = {'units':'nT',
                      'desc':'Total geomagnetic field from IGRF.'}
    inst.meta['B_east'] = {'units':'nT',
                           'desc':'Geomagnetic field from IGRF expressed using the East/North/Up (ENU) basis.'}
    inst.meta['B_north'] = {'units':'nT',
                            'desc':'Geomagnetic field from IGRF expressed using the East/North/Up (ENU) basis.'}
    inst.meta['B_up'] = {'units':'nT',
                         'desc':'Geomagnetic field from IGRF expressed using the East/North/Up (ENU) basis.'}

    inst.meta['B_ecef_x'] = {'units':'nT',
                             'desc':'Geomagnetic field from IGRF expressed using the Earth Centered Earth Fixed (ECEF) basis.'}
    inst.meta['B_ecef_y'] = {'units':'nT',
                             'desc':'Geomagnetic field from IGRF expressed using the Earth Centered Earth Fixed (ECEF) basis.'}
    inst.meta['B_ecef_z'] = {'units':'nT',
                             'desc':'Geomagnetic field from IGRF expressed using the Earth Centered Earth Fixed (ECEF) basis.'}
    return