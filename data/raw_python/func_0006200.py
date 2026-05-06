def get_fobj(fname, mode='w+'):
    """Obtain a proper file object.

    Parameters
    ----------
    fname : string, file object, file descriptor
        If a string or file descriptor, then we create a file object. If
        *fname* is a file object, then we do nothing and ignore the specified
        *mode* parameter.
    mode : str
        The mode of the file to be opened.

    Returns
    -------
    fobj : file object
        The file object.
    close : bool
        If *fname* was a string, then *close* will be *True* to signify that
        the file object should be closed after writing to it. Otherwise,
        *close* will be *False* signifying that the user, in essence,
        created the file object already and that subsequent operations
        should not close it.

    """
    if is_string_like(fname):
        fobj = open(fname, mode)
        close = True
    elif hasattr(fname, 'write'):
        # fname is a file-like object, perhaps a StringIO (for example)
        fobj = fname
        close = False
    else:
        # assume it is a file descriptor
        fobj = os.fdopen(fname, mode)
        close = False
    return fobj, close