def FixmatFactory(fixmatfile, categories = None, var_name = 'fixmat', field_name='x'):
    """
    Loads a single fixmat (fixmatfile).
    
    Parameters:
        fixmatfile : string
            The matlab fixmat that should be loaded.
        categories : instance of stimuli.Categories, optional
            Links data in categories to data in fixmat.
    """
    try:
        data = loadmat(fixmatfile, struct_as_record = False)
        keys = list(data.keys())
        data = data[var_name][0][0]
    except KeyError:
        raise RuntimeError('%s is not a field of the matlab structure. Possible'+
                'Keys are %s'%str(keys))
    
    num_fix = data.__getattribute__(field_name).size

    # Get a list with fieldnames and a list with parameters
    fields = {}
    parameters = {}
    for field in data._fieldnames:
        if data.__getattribute__(field).size == num_fix:
            fields[field] = data.__getattribute__(field)
        else:            
            parameters[field] = data.__getattribute__(field)[0].tolist()
            if len(parameters[field]) == 1:
                parameters[field] = parameters[field][0]
    
    # Generate FixMat
    fixmat = FixMat(categories = categories)
    fixmat._fields = list(fields.keys())
    for (field, value) in list(fields.items()):
        fixmat.__dict__[field] = value.reshape(-1,) 

    fixmat._parameters = parameters
    fixmat._subjects = None
    for (field, value) in list(parameters.items()):
        fixmat.__dict__[field] = value
    fixmat._num_fix = num_fix
    return fixmat