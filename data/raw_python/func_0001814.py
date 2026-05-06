def temp_dir(folder=None, delete=True):
    # type: (Optional[str], bool) -> str
    """Get a temporary directory optionally with folder appended (and created if it doesn't exist)

    Args:
        folder (Optional[str]): Folder to create in temporary folder. Defaults to None.
        delete (bool): Whether to delete folder on exiting with statement

    Returns:
        str: A temporary directory
    """
    tempdir = get_temp_dir()
    if folder:
        tempdir = join(tempdir, folder)
    if not exists(tempdir):
        makedirs(tempdir)
    try:
        yield tempdir
    finally:
        if delete:
            rmtree(tempdir)