def remove_empty_files(root_directory, dry_run=False, ignore_errors=True,
                       enable_scandir=False):
    """
    Remove all empty files from a path. Returns list of the empty files removed.

    :param root_directory: base directory to start at
    :param dry_run: just return a list of what would be removed
    :param ignore_errors: Permissions are a pain, just ignore if you blocked
    :param enable_scandir: on python < 3.5 enable external scandir package
    :return: list of removed files
    """
    file_list = []
    for root, directories, files in _walk(root_directory,
                                          enable_scandir=enable_scandir):
        for file_name in files:
            file_path = join_paths(root, file_name, strict=True)
            if os.path.isfile(file_path) and not os.path.getsize(file_path):
                if file_hash(file_path) == variables.hashes.empty_file.md5:
                    file_list.append(file_path)

    file_list = sorted(set(file_list))

    if not dry_run:
        for afile in file_list:
            try:
                os.unlink(afile)
            except OSError as err:
                if ignore_errors:
                    logger.info("File {0} could not be deleted".format(afile))
                else:
                    raise err

    return file_list