def dict_to_nvlist(dict):
    '''Convert a dictionary into a CORBA namevalue list.'''
    result = []
    for item in list(dict.keys()):
        result.append(SDOPackage.NameValue(item, omniORB.any.to_any(dict[item])))
    return result