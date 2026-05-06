def _get_last_dirs(path, num=1):
    """Get a path including only the trailing `num` directories.

    Returns
    -------
    last_path : str

    """
    head, tail = os.path.split(path)
    last_path = str(tail)
    for ii in range(num):
        head, tail = os.path.split(head)
        last_path = os.path.join(tail, last_path)

    last_path = "..." + last_path
    return last_path