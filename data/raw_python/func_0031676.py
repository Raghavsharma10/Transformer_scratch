def load_dict_of_nested_lists_from_h5(fname, toplevelkeys=None):
    """
    Load nested list structure from hdf5 file


    Parameters
    ----------        
    fname : str
        Filename
    toplevelkeys : None or iterable, 
        Load a two(default) or three-layered structure.


    Returns
    -------
    dict(list(numpy.ndarray))
        dictionary of nested lists with variable length array data.
    
    """
    
    # Container:
    data = {}

    # Open file object
    f = h5py.File(fname, 'r')

    # Iterate over partial dataset
    if toplevelkeys is not None:
        for i in toplevelkeys:
            ivalue = f[str(i)]
            data[i] = []
            for j, jvalue in enumerate(ivalue.values()):
                data[int(i)].append([])
                for k, kvalue in enumerate(jvalue.values()):
                    data[i][j].append(kvalue.value)
    else:
        for i, ivalue in list(f.items()):
            i = int(i)
            data[i] = []
            for j, jvalue in enumerate(ivalue.values()):
                data[i].append([])
                for k, kvalue in enumerate(jvalue.values()):
                    data[i][j].append(kvalue.value)

    # Close dataset
    f.close()

    return data