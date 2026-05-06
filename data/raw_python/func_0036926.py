def _setdefault_via_pathlist(external_dict,path_list,**kwargs):
    '''
        #if path_list already in external_dict, will do nothing
        y = {}
        path_list = ['c','b']
        _setdefault_via_pathlist(y,path_list)
        y
        _setdefault_via_pathlist(y,path_list)
        y = {}
        _setdefault_via_pathlist(y,path_list)
        y
    '''
    if('s2n' in kwargs):
        s2n = kwargs['s2n']
    else:
        s2n = 0
    if('n2s' in kwargs):
        n2s = kwargs['n2s']
    else:
        n2s = 0
    if('default_element' in kwargs):
        default_element = kwargs['default_element']
    else:
        default_element = {}
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
            this.__getitem__(key)
        except:
            try:
                # necessary ,when default_element = {} or []
                de = copy.deepcopy(default_element)
                this.__setitem__(key,de)
            except:
                return(external_dict)
            else:
                pass
            this = this.__getitem__(key)
        else:
            this = this.__getitem__(key)
    return(external_dict)