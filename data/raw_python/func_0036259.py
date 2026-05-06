def nvlist_to_dict(nvlist):
    '''Convert a CORBA namevalue list into a dictionary.'''
    result = {}
    for item in nvlist :
        result[item.name] = item.value.value()
    return result