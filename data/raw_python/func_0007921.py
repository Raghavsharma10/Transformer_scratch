def _aspectProperties(obj1, obj2, aspDict):
    """ Returns the properties of an aspect between
    obj1 and obj2, given by 'aspDict'. 
    
    This function assumes obj1 to be the active object, 
    i.e., the one responsible for starting the aspect.
    
    """
    orb = aspDict['orb']
    asp = aspDict['type']
    sep = aspDict['separation']
    
    # Properties
    prop1 = {
        'id': obj1.id,
        'inOrb': False,
        'movement': const.NO_MOVEMENT         
    }
    prop2 = {
        'id': obj2.id,
        'inOrb': False,
        'movement': const.NO_MOVEMENT         
    }
    prop = {
        'type': asp,
        'orb': orb,
        'direction': -1,
        'condition': -1,
        'active': prop1,
        'passive': prop2        
    }
    
    if asp == const.NO_ASPECT:
        return prop
    
    # Aspect within orb
    prop1['inOrb'] = orb <= obj1.orb()
    prop2['inOrb'] = orb <= obj2.orb()
    
    # Direction
    prop['direction'] = const.DEXTER if sep <= 0 else const.SINISTER
    
    # Sign conditions
    # Note: if obj1 is before obj2, orbDir will be less than zero
    orbDir = sep-asp if sep >= 0 else sep+asp
    offset = obj1.signlon + orbDir
    if 0 <= offset < 30:
        prop['condition'] = const.ASSOCIATE
    else:
        prop['condition'] = const.DISSOCIATE 
    
    # Movement of the individual objects
    if abs(orbDir) < MAX_EXACT_ORB:
        prop1['movement'] = prop2['movement'] = const.EXACT
    else:
        # Active object applies to Passive if it is before 
        # and direct, or after the Passive and Rx..
        prop1['movement'] = const.SEPARATIVE
        if (orbDir > 0 and obj1.isDirect()) or \
                (orbDir < 0 and obj1.isRetrograde()):
            prop1['movement'] = const.APPLICATIVE
        elif obj1.isStationary():
            prop1['movement'] = const.STATIONARY
        
        # The Passive applies or separates from the Active 
        # if it has a different direction..
        # Note: Non-planets have zero speed
        prop2['movement'] = const.NO_MOVEMENT
        obj2speed = obj2.lonspeed if obj2.isPlanet() else 0.0
        sameDir = obj1.lonspeed * obj2speed >= 0
        if not sameDir:
            prop2['movement'] = prop1['movement']
        
    return prop