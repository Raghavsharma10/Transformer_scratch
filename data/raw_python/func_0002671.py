def for_each_file(base_dir, func):
    """
    Calls func(filename) for every file under base_dir.

    :param base_dir: A directory containing files
    :param func: The function to call with every file.
    """

    for dir_path, _, file_names in os.walk(base_dir):
        for filename in file_names:
            func(os.path.join(dir_path, filename))