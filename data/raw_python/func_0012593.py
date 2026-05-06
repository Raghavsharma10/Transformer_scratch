def load(path, variable='Datamat'):
    """
    Load datamat at path.

    Parameters:
        path : string
            Absolute path of the file to load from.
    """
    f = h5py.File(path,'r')
    try:
        dm = fromhdf5(f[variable])
    finally:
        f.close()
    return dm