def getModifiers(chart):
    """ Returns the factors of the temperament modifiers. """
    
    modifiers = []
    
    # Factors which can be affected
    asc = chart.getAngle(const.ASC)
    ascRulerID = essential.ruler(asc.sign)
    ascRuler = chart.getObject(ascRulerID)
    moon = chart.getObject(const.MOON)
    factors = [
        [MOD_ASC, asc],
        [MOD_ASC_RULER, ascRuler],
        [MOD_MOON, moon]
    ]
    
    # Factors of affliction
    mars = chart.getObject(const.MARS)
    saturn = chart.getObject(const.SATURN)
    sun = chart.getObject(const.SUN)
    affect = [
        [mars, [0, 90, 180]],
        [saturn, [0, 90, 180]],
        [sun, [0]]     
    ]
    
    # Do calculations of afflictions
    for affectingObj, affectingAsps in affect:
        for factor, affectedObj in factors:
            modf = modifierFactor(chart, 
                                  factor, 
                                  affectedObj, 
                                  affectingObj, 
                                  affectingAsps)
            if modf:
                modifiers.append(modf)
    
    return modifiers