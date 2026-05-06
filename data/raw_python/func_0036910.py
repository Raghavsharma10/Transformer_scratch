def tlist2dict(tuple_list,**kwargs):
    '''
        #duplicate keys will lost
        tl = [(1,2),(3,4),(1,5)]
        tlist2dict(tl)
    '''
    if('deepcopy' in kwargs):
        deepcopy = kwargs['deepcopy']
    else:
        deepcopy = 1
    if('check' in kwargs):
        check = kwargs['check']
    else:
        check = 1
    if(check):
        if(tltl.is_tlist(tuple_list)):
            pass
        else:
            return(None)
    else:
        pass
    j = {}
    if(deepcopy):
        new = copy.deepcopy(tuple_list)
    else:
        new = tuple_list
    for i in range(0,new.__len__()):
        temp = new[i]
        key = temp[0]
        value = temp[1]
        j[key] = value
    return(j)