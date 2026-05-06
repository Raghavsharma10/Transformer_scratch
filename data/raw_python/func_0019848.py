def load_fegg(filepath, update=True):
    """
    Loads pickled egg

    Parameters
    ----------
    filepath : str
        Location of pickled egg

    update : bool
        If true, updates egg to latest format

    Returns
    ----------
    egg : Egg data object
        A loaded unpickled egg

    """
    try:
        egg = FriedEgg(**dd.io.load(filepath))
    except ValueError as e:
        print(e)
        # if error, try loading old format
        with open(filepath, 'rb') as f:
            egg = pickle.load(f)

    if update:
        return egg.crack()
    else:
        return egg