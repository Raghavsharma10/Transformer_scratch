def getFactors(chart):
    """ Returns the factors for the temperament. """
    
    factors = []
    
    # Asc sign
    asc = chart.getAngle(const.ASC)
    singleFactor(factors, chart, ASC_SIGN, asc.sign)
    
    # Asc ruler
    ascRulerID = essential.ruler(asc.sign)
    ascRuler = chart.getObject(ascRulerID)
    singleFactor(factors, chart, ASC_RULER, ascRuler)
    singleFactor(factors, chart, ASC_RULER_SIGN, ascRuler.sign)
    
    # Planets in House 1
    house1 = chart.getHouse(const.HOUSE1)
    planetsHouse1 = chart.objects.getObjectsInHouse(house1)
    for obj in planetsHouse1:
        singleFactor(factors, chart, HOUSE1_PLANETS_IN, obj)
        
    # Planets conjunct Asc
    planetsConjAsc = chart.objects.getObjectsAspecting(asc, [0])
    for obj in planetsConjAsc:
        # Ignore planets already in house 1
        if obj not in planetsHouse1:
            singleFactor(factors, chart, ASC_PLANETS_CONJ, obj)
            
    # Planets aspecting Asc cusp
    aspList = [60, 90, 120, 180]
    planetsAspAsc = chart.objects.getObjectsAspecting(asc, aspList)
    for obj in planetsAspAsc:
        aspect = aspects.aspectType(obj, asc, aspList)
        singleFactor(factors, chart, ASC_PLANETS_ASP, obj, aspect)
    
    # Moon sign and phase
    moon = chart.getObject(const.MOON)
    singleFactor(factors, chart, MOON_SIGN, moon.sign)
    singleFactor(factors, chart, MOON_PHASE, moon)
    
    # Moon dispositor
    moonRulerID = essential.ruler(moon.sign)
    moonRuler = chart.getObject(moonRulerID)
    moonFactor = singleFactor(factors, chart, MOON_DISPOSITOR_SIGN, moonRuler.sign)
    moonFactor['planetID'] = moonRulerID  # Append moon dispositor ID
    
    # Planets conjunct Moon
    planetsConjMoon = chart.objects.getObjectsAspecting(moon, [0])
    for obj in planetsConjMoon:
        singleFactor(factors, chart, MOON_PLANETS_CONJ, obj)
            
    # Planets aspecting Moon
    aspList = [60, 90, 120, 180]
    planetsAspMoon = chart.objects.getObjectsAspecting(moon, aspList)
    for obj in planetsAspMoon:
        aspect = aspects.aspectType(obj, moon, aspList)
        singleFactor(factors, chart, MOON_PLANETS_ASP, obj, aspect)
    
    # Sun season
    sun = chart.getObject(const.SUN)
    singleFactor(factors, chart, SUN_SEASON, sun)
    
    return factors