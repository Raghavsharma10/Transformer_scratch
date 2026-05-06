def load(filepath, update=True):
    """
    Loads eggs, fried eggs ands example data

    Parameters
    ----------
    filepath : str
        Location of file

    update : bool
        If true, updates egg to latest format

    Returns
    ----------
    data : quail.Egg or quail.FriedEgg
        Data loaded from disk

    """

    if filepath == 'automatic' or filepath == 'example':
        fpath = os.path.dirname(os.path.abspath(__file__)) + '/data/automatic.egg'
        return load_egg(fpath)
    elif filepath == 'manual':
        fpath = os.path.dirname(os.path.abspath(__file__)) + '/data/manual.egg'
        return load_egg(fpath, update=False)
    elif filepath == 'naturalistic':
        fpath = os.path.dirname(os.path.abspath(__file__)) + '/data/naturalistic.egg'
    elif filepath.split('.')[-1]=='egg':
        return load_egg(filepath, update=update)
    elif filepath.split('.')[-1]=='fegg':
        return load_fegg(filepath, update=False)
    else:
        raise ValueError('Could not load file.')