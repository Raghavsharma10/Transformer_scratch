def load(path):
    """
    Load fixmat at path.
    
    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    if 'Fixmat' in f:
      fm_group = f['Fixmat']
    else:
      fm_group = f['Datamat']
    fields = {}
    params = {}
    for field, value in list(fm_group.items()):
        fields[field] = np.array(value)
    for param, value in list(fm_group.attrs.items()):
        params[param] = value
    f.close()
    return VectorFixmatFactory(fields, params)