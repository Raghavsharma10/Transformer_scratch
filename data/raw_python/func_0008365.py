def check_dirs():
    """
    Check the directories if they exist.

    :raises FileExistsError: if a file exists but is not a directory
    """
    dirs = [MAIN_DIR, TEMP_DIR, DOWNLOAD_DIR, SAVESTAT_DIR]
    for directory in dirs:
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(str(directory.resolve()) + " cannot be used as a directory.")