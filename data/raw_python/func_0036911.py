def dict2tlist(this_dict,**kwargs):
    '''
        #sequence will be losted
        d = {'a':'b','c':'d'}
        dict2tlist(d)
    '''
    if('check' in kwargs):
        check = kwargs['check']
    else:
        check = 1
    if(check):
        if(isinstance(this_dict,dict)):
            pass
        else:
            return(None)
    else:
        pass
    if('deepcopy' in kwargs):
        deepcopy = kwargs['deepcopy']
    else:
        deepcopy = 1
    tuple_list = []
    if(deepcopy):
        new = copy.deepcopy(this_dict)
    else:
        new = this_dict
    i = 0
    for key in this_dict:
        value = this_dict[key]
        tuple_list.append((key,value))
    return(tuple_list)