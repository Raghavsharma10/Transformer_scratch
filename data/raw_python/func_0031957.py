def dict_of_numpyarray_to_dict_of_list(d):
    '''
    Convert dictionary containing numpy arrays to dictionary containing lists
    
    Parameters
    ----------
    d : dict
        sli parameter name and value as dictionary key and value pairs
    
    Returns
    -------
    d : dict
        modified dictionary
    
    '''
    for key,value in d.iteritems():
        if isinstance(value,dict):  # if value == dict 
            # recurse
            d[key] = dict_of_numpyarray_to_dict_of_list(value)
        elif isinstance(value,np.ndarray): # or isinstance(value,list) :
            d[key] = value.tolist()
    return d