def _diff(d1,d2):
    '''
        d1 = {'a':'x','b':'y','c':'z'}
        d2 = {'a':'x','b':'u','d':'v'}
        _diff(d1,d2)
        _diff(d2,d1)
    '''
    d = {}
    ds = _diff_internal(d1,d2)
    for key in ds['vdiff']:
        d[key] = d1[key]
    for key in ds['kdiff']:
        d[key] = d1[key]
    return(d)