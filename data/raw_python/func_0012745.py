def rvalue(ddtt):
    """
    R value (W/K) of a construction or material.
    thickness (m) / conductivity (W/m-K)
    """
    object_type = ddtt.obj[0]
    if object_type == 'Construction':
        rvalue = INSIDE_FILM_R + OUTSIDE_FILM_R
        layers = ddtt.obj[2:]
        field_idd = ddtt.getfieldidd('Outside_Layer')
        validobjects = field_idd['validobjects']
        for layer in layers:
            found = False
            for key in validobjects:
                try:
                    rvalue += ddtt.theidf.getobject(key, layer).rvalue
                    found = True
                except AttributeError:
                    pass
            if not found:
                raise AttributeError("%s material not found in IDF" % layer)
    elif object_type == 'Material':
        thickness = ddtt.obj[ddtt.objls.index('Thickness')]
        conductivity = ddtt.obj[ddtt.objls.index('Conductivity')]
        rvalue = thickness / conductivity
    elif object_type == 'Material:AirGap':
        rvalue = ddtt.obj[ddtt.objls.index('Thermal_Resistance')] 
    elif object_type == 'Material:InfraredTransparent':
        rvalue = 0
    elif object_type == 'Material:NoMass':
        rvalue = ddtt.obj[ddtt.objls.index('Thermal_Resistance')] 
    elif object_type == 'Material:RoofVegetation':
        warnings.warn(
            "Material:RoofVegetation thermal properties are based on dry soil",
            UserWarning)
        thickness = ddtt.obj[ddtt.objls.index('Thickness')]
        conductivity = ddtt.obj[ddtt.objls.index('Conductivity_of_Dry_Soil')]
        rvalue = thickness / conductivity
    else:
        raise AttributeError("%s rvalue property not implemented" % object_type)
    return rvalue