def load_egg(filepath, update=True):
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
        egg = Egg(**dd.io.load(filepath))
    except:
        # if error, try loading old format
        with open(filepath, 'rb') as f:
            egg = pickle.load(f)

    if update:
        if egg.meta:
            old_meta = egg.meta
            egg.crack()
            egg.meta = old_meta
            return egg
        else:
            return egg.crack()
    else:
        return egg