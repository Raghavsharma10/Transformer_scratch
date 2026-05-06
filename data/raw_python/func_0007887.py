def sweNextTransit(obj, jd, lat, lon, flag):
    """ Returns the julian date of the next transit of
    an object. The flag should be 'RISE' or 'SET'. 
    
    """
    sweObj = SWE_OBJECTS[obj]
    flag = swisseph.CALC_RISE if flag == 'RISE' else swisseph.CALC_SET
    trans = swisseph.rise_trans(jd, sweObj, lon, lat, 0, 0, 0, flag)
    return trans[1][0]