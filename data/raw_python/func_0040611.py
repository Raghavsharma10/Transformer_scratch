def remove_file(paths):
    """
    Remove file from paths introduced.
    """

    for path in force_list(paths):
        if os.path.exists(path):
            os.remove(path)