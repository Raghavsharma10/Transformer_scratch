def _keys_via_value_nonrecur(d,v):
    '''
        #non-recursive
        d = {1:'a',2:'b',3:'a'}
        _keys_via_value_nonrecur(d,'a')
    '''
    rslt = []
    for key in d:
        if(d[key] == v):
            rslt.append(key)
    return(rslt)