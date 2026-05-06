def haiz(obj, chart):
    """ Returns if an object is in Haiz. """
    objGender = obj.gender()
    objFaction = obj.faction()
    
    if obj.id == const.MERCURY:
        # Gender and faction of mercury depends on orientality
        sun = chart.getObject(const.SUN)
        orientalityM = orientality(obj, sun)
        if orientalityM == ORIENTAL:
            objGender = const.MASCULINE
            objFaction = const.DIURNAL
        else:
            objGender = const.FEMININE
            objFaction = const.NOCTURNAL
            
    # Object gender match sign gender?
    signGender = props.sign.gender[obj.sign]
    genderConformity = (objGender == signGender)
    
    # Match faction
    factionConformity = False
    diurnalChart = chart.isDiurnal()
    
    if obj.id == const.SUN and not diurnalChart:
        # Sun is in conformity only when above horizon
        factionConformity = False
    else:
        # Get list of houses in the chart's diurnal faction
        if diurnalChart:
            diurnalFaction = props.house.aboveHorizon
            nocturnalFaction = props.house.belowHorizon
        else:
            diurnalFaction = props.house.belowHorizon
            nocturnalFaction = props.house.aboveHorizon
        
        # Get the object's house and match factions
        objHouse = chart.houses.getObjectHouse(obj)
        if (objFaction == const.DIURNAL and objHouse.id in diurnalFaction or
            objFaction == const.NOCTURNAL and objHouse.id in nocturnalFaction):
                factionConformity = True
        
    # Match things
    if (genderConformity and factionConformity):
        return HAIZ
    elif (not genderConformity and not factionConformity):
        return CHAIZ
    else:
        return None