def is_leaf(obj):
    '''
        the below is for nested-dict
        any type is not dict will be treated as a leaf
        empty dict will be treated as a leaf
        from edict.edict import *
        is_leaf(1)
        is_leaf({1:2})
        is_leaf({})
    '''
    if(is_dict(obj)):
        length = obj.__len__()
        if(length == 0):
            return(True)
        else:
            return(False)
    else:
        return(True)