def _include_pathlist(external_dict,path_list,**kwargs):
    '''
        y = {
            'a':
                {'x':88},
            'b':
                {
                    'x':
                        {'c':66}
                }
        }
        _include_pathlist(y,['a'])
        _include_pathlist(y,['a','x'])
        _include_pathlist(y,['b','x','c'])
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
    for i in range(0,path_list.__len__()):
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
        try:
            this = this.__getitem__(key)
        except:
            return(False)
        else:
            pass
    return(True)