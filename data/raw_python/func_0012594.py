def VectorFactory(fields, parameters, categories = None):
    '''
    Creates a datamat from a dictionary that contains lists/arrays as values.

    Input:
        fields: Dictionary
            The values will be used as fields of the datamat and the keys
            as field names.
        parameters: Dictionary
            A dictionary whose values are added as parameters. Keys are used
            for parameter names.
    '''
    fm = Datamat(categories = categories)
    fm._fields = list(fields.keys())
    for (field, value) in list(fields.items()):
        try:
            fm.__dict__[field] = np.asarray(value)
        except ValueError:
            fm.__dict__[field] = np.asarray(value, dtype=np.object)

    fm._parameters = parameters
    for (field, value) in list(parameters.items()):
       fm.__dict__[field] = value
    fm._num_fix = len(fm.__dict__[list(fields.keys())[0]])
    return fm