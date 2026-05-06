def ExistingFileOrNone(fname):
    """Like `Existingfile`, but if `fname` is string "None" then return `None`."""
    if os.path.isfile(fname):
        return fname
    elif fname.lower() == 'none':
        return None
    else:
        raise ValueError("%s must specify a valid file name or 'None'" % fname)