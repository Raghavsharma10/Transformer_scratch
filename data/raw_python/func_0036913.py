def dict_mirror(d,**kwargs):
    '''
        d = {1:'a',2:'a',3:'b'}
    '''
    md = {}
    if('sort_func' in kwargs):
        sort_func = kwargs['sort_func']
    else:
        sort_func = sorted
    vl = list(d.values())
    uvl = elel.uniqualize(vl)
    for v in uvl:
        kl = _keys_via_value_nonrecur(d,v)
        k = sorted(kl)[0]
        md[v] = k
    return(md)