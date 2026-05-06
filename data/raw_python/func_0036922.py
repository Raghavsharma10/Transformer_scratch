def _extend(dict1,dict2,**kwargs):
    '''
        dict1 = {1:'a',2:'b',3:'c',4:'d'}
        dict2 = {5:'u',2:'v',3:'w',6:'x',7:'y'}
        d = _extend(dict1,dict2)
        pobj(d)
        dict1 = {1:'a',2:'b',3:'c',4:'d'}
        dict2 = {5:'u',2:'v',3:'w',6:'x',7:'y'}
        d = _extend(dict1,dict2,overwrite=1)
        pobj(d)
    '''
    if('deepcopy' in kwargs):
        deepcopy=kwargs['deepcopy']
    else:
        deepcopy=1
    if('overwrite' in kwargs):
        overwrite=kwargs['overwrite']
    else:
        overwrite=0
    if(deepcopy):
        dict1 = copy.deepcopy(dict1)
        dict2 = copy.deepcopy(dict2)
    else:
        pass
    d = dict1
    for key in dict2:
        if(key in dict1):
            if(overwrite):
                d[key] = dict2[key]
            else:
                pass
        else:
            d[key] = dict2[key]
    return(d)