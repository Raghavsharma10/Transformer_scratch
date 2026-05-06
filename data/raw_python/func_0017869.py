def _split_path(path):
    """
    A wrapper around the normal split function that ignores any trailing /.

    :return: A tuple of the form (dirname, last) where last is the last element
             in the path.
    """
    # Get around a quirk in path_split where a / at the end will make the
    # dirname (split[0]) the entire path
    path = path[:-1] if path[-1] == '/' else path
    split = path_split(path)
    return split