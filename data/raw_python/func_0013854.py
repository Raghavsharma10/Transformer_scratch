def add_iri_thermal_plasma(inst, glat_label='glat', glong_label='glong', 
                                 alt_label='alt'):
    """ 
    Uses IRI (International Reference Ionosphere) model to simulate an ionosphere.
    
    Uses pyglow module to run IRI. Configured to use actual solar parameters to run 
    model.
    
    Example
    -------
        # function added velow modifies the inst object upon every inst.load call
        inst.custom.add(add_iri_thermal_plasma, 'modify', glat_label='custom_label')
    
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
        Input pysat.Instrument object modified to include thermal plasma parameters.
        'ion_temp' for ion temperature in Kelvin
        'e_temp' for electron temperature in Kelvin
        'ion_dens' for the total ion density (O+ and H+)
        'frac_dens_o' for the fraction of total density that is O+
        'frac_dens_h' for the fraction of total density that is H+
        
    """

    import pyglow
    from pyglow.pyglow import Point
    
    iri_params = []
    # print 'IRI Simulations'
    for time,lat,lon,alt in zip(inst.data.index, inst[glat_label], inst[glong_label], inst[alt_label]):
        # Point class is instantiated. Its parameters are a function of time and spatial location
        pt = Point(time,lat,lon,alt)
        pt.run_iri()
        iri = {}
        # After the model is run, its members like Ti, ni[O+], etc. can be accessed
        iri['ion_temp'] = pt.Ti
        iri['e_temp'] = pt.Te
        iri['ion_dens'] = pt.ni['O+'] + pt.ni['H+'] + pt.ni['HE+']#pt.ne - pt.ni['NO+'] - pt.ni['O2+'] - pt.ni['HE+']
        iri['frac_dens_o'] = pt.ni['O+']/iri['ion_dens']
        iri['frac_dens_h'] = pt.ni['H+']/iri['ion_dens']
        iri['frac_dens_he'] = pt.ni['HE+']/iri['ion_dens']
        iri_params.append(iri)        
    # print 'Complete.'
    iri = pds.DataFrame(iri_params)
    iri.index = inst.data.index
    inst[iri.keys()] = iri
    
    inst.meta['ion_temp'] = {'units':'Kelvin','long_name':'Ion Temperature'}
    inst.meta['ion_dens'] = {'units':'N/cc','long_name':'Ion Density',
                             'desc':'Total ion density including O+ and H+ from IRI model run.'}
    inst.meta['frac_dens_o'] = {'units':'','long_name':'Fractional O+ Density'}
    inst.meta['frac_dens_h'] = {'units':'','long_name':'Fractional H+ Density'}