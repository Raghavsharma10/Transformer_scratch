def _union(d1,d2):
    '''
        d1 = {'a':'x','b':'y','c':'z'}
        d2 = {'a':'x','b':'u','d':'v'}
        _union(d1,d2)
        _union(d2,d1)
    '''
    u = {}
    ds = _diff_internal(d1,d2)
    for key in ds['same']:
        u[key] = d1[key]
    for key in ds['vdiff']:
        u[key] = d1[key]
    for key in ds['kdiff']:
        u[key] = d1[key]
    ds = _diff_internal(d2,d1)
    for key in ds['kdiff']:
        u[key] = d2[key]
    return(u)