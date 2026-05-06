def heatcapacity(ddtt):
    """
    Heat capacity (kJ/m2-K) of a construction or material.
    thickness (m) * density (kg/m3) * specific heat (J/kg-K) * 0.001
    """
    object_type = ddtt.obj[0]
    if object_type == 'Construction':
        heatcapacity = 0
        layers = ddtt.obj[2:]
        field_idd = ddtt.getfieldidd('Outside_Layer')
        validobjects = field_idd['validobjects']
        for layer in layers:
            found = False
            for key in validobjects:
                try:
                    heatcapacity += ddtt.theidf.getobject(key, layer).heatcapacity
                    found = True
                except AttributeError:
                    pass
            if not found:
                raise AttributeError("%s material not found in IDF" % layer)
    elif object_type == 'Material':
        thickness = ddtt.obj[ddtt.objls.index('Thickness')]
        density = ddtt.obj[ddtt.objls.index('Density')]
        specificheat = ddtt.obj[ddtt.objls.index('Specific_Heat')]
        heatcapacity = thickness * density * specificheat * 0.001
    elif object_type == 'Material:AirGap':
        heatcapacity = 0
    elif object_type == 'Material:InfraredTransparent':
        heatcapacity = 0
    elif object_type == 'Material:NoMass':
        warnings.warn(
            "Material:NoMass materials included in heat capacity calculation",
            UserWarning)
        heatcapacity = 0
    elif object_type == 'Material:RoofVegetation':
        warnings.warn(
            "Material:RoofVegetation thermal properties are based on dry soil",
            UserWarning)
        thickness = ddtt.obj[ddtt.objls.index('Thickness')]
        density = ddtt.obj[ddtt.objls.index('Density_of_Dry_Soil')]
        specificheat = ddtt.obj[ddtt.objls.index('Specific_Heat_of_Dry_Soil')]
        heatcapacity = thickness * density * specificheat * 0.001
    else:
        raise AttributeError("%s has no heatcapacity property" % object_type)
    return heatcapacity