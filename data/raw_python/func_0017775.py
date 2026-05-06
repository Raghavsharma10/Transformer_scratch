def get_data_path(filename):
    """
    Get the path of the given file within the batchup data directory

    Parameters
    ----------
    filename: str
        The filename to locate within the batchup data directory

    Returns
    -------
    str
        The full path of the file
    """
    if os.path.isabs(filename):
        return filename
    else:
        return os.path.join(get_data_dir(), filename)