def dup_finder(file_path, directory=".", enable_scandir=False):
    """
    Check a directory for duplicates of the specified file. This is meant
    for a single file only, for checking a directory for dups, use
    directory_duplicates.

    This is designed to be as fast as possible by doing lighter checks
    before progressing to
    more extensive ones, in order they are:

    1. File size
    2. First twenty bytes
    3. Full SHA256 compare

    .. code:: python

        list(reusables.dup_finder(
             "test_structure\\files_2\\empty_file"))
        # ['C:\\Reusables\\test\\data\\fake_dir',
        #  'C:\\Reusables\\test\\data\\test_structure\\Files\\empty_file_1',
        #  'C:\\Reusables\\test\\data\\test_structure\\Files\\empty_file_2',
        #  'C:\\Reusables\\test\\data\\test_structure\\files_2\\empty_file']

    :param file_path: Path to file to check for duplicates of
    :param directory: Directory to dig recursively into to look for duplicates
    :param enable_scandir: on python < 3.5 enable external scandir package
    :return: generators
    """
    size = os.path.getsize(file_path)
    if size == 0:
        for empty_file in remove_empty_files(directory, dry_run=True):
            yield empty_file
    else:
        with open(file_path, 'rb') as f:
            first_twenty = f.read(20)
        file_sha256 = file_hash(file_path, "sha256")

        for root, directories, files in _walk(directory,
                                              enable_scandir=enable_scandir):
            for each_file in files:
                test_file = os.path.join(root, each_file)
                if os.path.getsize(test_file) == size:
                    try:
                        with open(test_file, 'rb') as f:
                            test_first_twenty = f.read(20)
                    except OSError:
                        logger.warning("Could not open file to compare - "
                                       "{0}".format(test_file))
                    else:
                        if first_twenty == test_first_twenty:
                            if file_hash(test_file, "sha256") == file_sha256:
                                yield os.path.abspath(test_file)