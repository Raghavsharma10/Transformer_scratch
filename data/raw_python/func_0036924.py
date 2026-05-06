def _update_intersection(dict1,dict2,**kwargs):
    '''
        dict1 = {1:'a',2:'b',3:'c',4:'d'}
        dict2 = {5:'u',2:'v',3:'w',6:'x',7:'y'}
        _update_intersection(dict1,dict2)
        pobj(dict1)
        pobj(dict2)
    '''
    if('deepcopy' in kwargs):
        deepcopy = kwargs['deepcopy']
    else:
        deepcopy = 1
    if(deepcopy == 1):
        dict1 = copy.deepcopy(dict1)
    else:
        pass
    for key in dict2:
        if(key in dict1):
            dict1[key] = dict2[key]
    return(dict1)