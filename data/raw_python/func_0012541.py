def remove_empty_directories(root_directory, dry_run=False, ignore_errors=True,
                             enable_scandir=False):
    """
    Remove all empty folders from a path. Returns list of empty directories.

    :param root_directory: base directory to start at
    :param dry_run: just return a list of what would be removed
    :param ignore_errors: Permissions are a pain, just ignore if you blocked
    :param enable_scandir: on python < 3.5 enable external scandir package
    :return: list of removed directories
    """
    listdir = os.listdir
    if python_version < (3, 5) and enable_scandir:
        import scandir as _scandir

        def listdir(directory):
            return list(_scandir.scandir(directory))

    directory_list = []
    for root, directories, files in _walk(root_directory,
                                          enable_scandir=enable_scandir,
                                          topdown=False):
        if (not directories and not files and os.path.exists(root) and
                    root != root_directory and os.path.isdir(root)):
            directory_list.append(root)
            if not dry_run:
                try:
                    os.rmdir(root)
                except OSError as err:
                    if ignore_errors:
                        logger.info("{0} could not be deleted".format(root))
                    else:
                        raise err
        elif directories and not files:
            for directory in directories:
                directory = join_paths(root, directory, strict=True)
                if (os.path.exists(directory) and os.path.isdir(directory) and
                        not listdir(directory)):
                    directory_list.append(directory)
                    if not dry_run:
                        try:
                            os.rmdir(directory)
                        except OSError as err:
                            if ignore_errors:
                                logger.info("{0} could not be deleted".format(
                                    directory))
                            else:
                                raise err
    return directory_list