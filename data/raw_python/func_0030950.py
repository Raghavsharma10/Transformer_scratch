def make_sure_dir_exists(dir_, create_subfolders=False):
    """Ensures that a directory exists.

    Adapted from StackOverflow users "Bengt" and "Heikki Toivonen"
    (http://stackoverflow.com/a/5032238).

    Parameters
    ----------
    dir_: str
        The directory path.
    create_subfolders: bool, optional
        Whether to create any inexistent subfolders. [False]
    
    Returns
    -------
    None

    Raises
    ------
    OSError
        If a file system error occurs.
    """
    assert isinstance(dir_, (str, _oldstr))
    assert isinstance(create_subfolders, bool)

    try:
        if create_subfolders:
            os.makedirs(dir_)
        else:
            os.mkdir(dir_)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise