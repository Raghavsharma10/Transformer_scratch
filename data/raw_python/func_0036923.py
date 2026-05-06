def _comprise(dict1,dict2):
    '''
        dict1 = {'a':1,'b':2,'c':3,'d':4}
        dict2 = {'b':2,'c':3}
        _comprise(dict1,dict2)
    '''
    len_1 = dict1.__len__()
    len_2 = dict2.__len__()
    if(len_2>len_1):
        return(False)
    else:
        for k2 in dict2:
            v2 = dict2[k2]
            if(k2 in dict1):
                v1 = dict1[k2]
                if(v1 == v2):
                    return(True)
                else:
                    return(False)
            else:
                return(False)