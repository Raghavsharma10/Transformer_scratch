def is_valid(file_path):
    """
    Check to see if a file exists or is empty.
    """
    from os import path, stat

    can_open = False

    try:
        with open(file_path) as fp:
            can_open = True
    except IOError:
        return False

    is_file = path.isfile(file_path)

    return path.exists(file_path) and is_file and stat(file_path).st_size > 0