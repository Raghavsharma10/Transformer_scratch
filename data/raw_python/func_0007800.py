def singleFactor(factors, chart, factor, obj, aspect=None):
    """" Single factor for the table. """
    
    objID = obj if type(obj) == str else obj.id
    res = {
        'factor': factor,
        'objID': objID,
        'aspect': aspect
    }
    
    # For signs (obj as string) return sign element
    if type(obj) == str:
        res['element'] = props.sign.element[obj]
        
    # For Sun return sign and sunseason element
    elif objID == const.SUN:
        sunseason = props.sign.sunseason[obj.sign]
        res['sign'] = obj.sign
        res['sunseason'] = sunseason
        res['element'] = props.base.sunseasonElement[sunseason]
        
    # For Moon return phase and phase element
    elif objID == const.MOON:
        phase = chart.getMoonPhase()
        res['phase'] = phase
        res['element'] = props.base.moonphaseElement[phase]
        
    # For regular planets return element or sign/sign element
    # if there's an aspect involved
    elif objID in const.LIST_SEVEN_PLANETS:
        if aspect:
            res['sign'] = obj.sign
            res['element'] = props.sign.element[obj.sign]
        else:
            res['element'] = obj.element()
            
    try:
        # If there's element, insert into list
        res['element']
        factors.append(res)
    except KeyError:
        pass
    
    return res