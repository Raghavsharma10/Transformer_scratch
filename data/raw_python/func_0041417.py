def flatten(nested_list):
    '''converts a list-of-lists to a single flat list'''
    return_list = []
    for i in nested_list:
        if isinstance(i,list):
            return_list += flatten(i)
        else:
            return_list.append(i)
    return return_list