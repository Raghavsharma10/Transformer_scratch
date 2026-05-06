def _copy_files(source, target):
    """
    Copy all the files in source directory to target.

    Ignores subdirectories.
    """
    source_files = listdir(source)
    if not exists(target):
        makedirs(target)
    for filename in source_files:
        full_filename = join(source, filename)
        if isfile(full_filename):
            shutil.copy(full_filename, target)