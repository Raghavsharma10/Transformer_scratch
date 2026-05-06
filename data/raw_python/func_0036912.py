def is_mirrable(d):
    '''
        d = {1:'a',2:'a',3:'b'}
    '''
    vl = list(d.values())
    lngth1 = vl.__len__()
    uvl = elel.uniqualize(vl)
    lngth2 = uvl.__len__()
    cond = (lngth1 == lngth2)
    return(cond)