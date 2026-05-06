def hourTable(date, pos):
    """ Creates the planetary hour table for a date 
    and position. 
    
    The table includes both diurnal and nocturnal 
    hour sequences and each of the 24 entries (12 * 2)
    are like (startJD, endJD, ruler).
    
    """
    
    lastSunrise = ephem.lastSunrise(date, pos)
    middleSunset = ephem.nextSunset(lastSunrise, pos)
    nextSunrise = ephem.nextSunrise(date, pos)
    table = []
    
    # Create diurnal hour sequence
    length = (middleSunset.jd - lastSunrise.jd) / 12.0
    for i in range(12):
        start = lastSunrise.jd + i * length
        end = start + length
        ruler = nthRuler(i, lastSunrise.date.dayofweek())
        table.append([start, end, ruler])
        
    # Create nocturnal hour sequence
    length = (nextSunrise.jd - middleSunset.jd) / 12.0
    for i in range(12):
        start = middleSunset.jd + i * length
        end = start + length
        ruler = nthRuler(i + 12, lastSunrise.date.dayofweek())
        table.append([start, end, ruler])
        
    return table