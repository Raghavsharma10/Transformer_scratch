def _complement(d1,d2):
    '''
        d1 = {'a':'x','b':'y','c':'z'}
        d2 = {'a':'x','b':'u','d':'v'}
        complement(d1,d2)
        complement(d2,d1)
    '''
    u = _union(d1,d2)
    c = _diff(u,d1)
    return(c)