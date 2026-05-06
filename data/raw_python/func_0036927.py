def _setitem_via_pathlist(external_dict,path_list,value,**kwargs):
    '''
        y = {'c': {'b': {}}}
        _setitem_via_pathlist(y,['c','b'],200)
    '''
    if('s2n' in kwargs):
        s2n = kwargs['s2n']
    else:
        s2n = 0
    if('n2s' in kwargs):
        n2s = kwargs['n2s']
    else:
        n2s = 0
    this = external_dict
    for i in range(0,path_list.__len__()-1):
        key = path_list[i]
        if(n2s ==1):
            key = str(key)
        if(s2n==1):
            try:
                int(key)
            except:
                pass
            else:
                key = int(key)
        this = this.__getitem__(key)
    this.__setitem__(path_list[-1],value)
    return(external_dict)