def scores(factors):
    """ Computes the score of temperaments
    and elements.
    
    """
    temperaments = {
        const.CHOLERIC: 0,
        const.MELANCHOLIC: 0,
        const.SANGUINE: 0,
        const.PHLEGMATIC: 0                
    }
    
    qualities = {
        const.HOT: 0,
        const.COLD: 0,
        const.DRY: 0,
        const.HUMID: 0
    }
    
    for factor in factors:
        element = factor['element']
        
        # Score temperament
        temperament = props.base.elementTemperament[element]
        temperaments[temperament] += 1
        
        # Score qualities
        tqualities = props.base.temperamentQuality[temperament]
        qualities[tqualities[0]] += 1
        qualities[tqualities[1]] += 1
        
    return {
        'temperaments': temperaments,
        'qualities': qualities
    }