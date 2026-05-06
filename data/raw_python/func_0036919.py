def _intersection(d1,d2):
    '''
        d1 = {'a':'x','b':'y','c':'z'}
        d2 = {'a':'x','b':'u','d':'v'}
        _intersection(d1,d2)
        _intersection(d2,d1)
    '''
    i = {}
    ds = _diff_internal(d1,d2)
    for key in ds['same']:
        i[key] = d1[key]
    return(i)