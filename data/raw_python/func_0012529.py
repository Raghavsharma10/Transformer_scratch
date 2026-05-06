def extract(archive_file, path=".", delete_on_success=False,
            enable_rar=False):
    """
    Automatically detect archive type and extract all files to specified path.

    .. code:: python

        import os

        os.listdir(".")
        # ['test_structure.zip']

        reusables.extract("test_structure.zip")

        os.listdir(".")
        # [ 'test_structure', 'test_structure.zip']


    :param archive_file: path to file to extract
    :param path: location to extract to
    :param delete_on_success: Will delete the original archive if set to True
    :param enable_rar: include the rarfile import and extract
    :return: path to extracted files
    """

    if not os.path.exists(archive_file) or not os.path.getsize(archive_file):
        logger.error("File {0} unextractable".format(archive_file))
        raise OSError("File does not exist or has zero size")

    arch = None
    if zipfile.is_zipfile(archive_file):
        logger.debug("File {0} detected as a zip file".format(archive_file))
        arch = zipfile.ZipFile(archive_file)
    elif tarfile.is_tarfile(archive_file):
        logger.debug("File {0} detected as a tar file".format(archive_file))
        arch = tarfile.open(archive_file)
    elif enable_rar:
        import rarfile
        if rarfile.is_rarfile(archive_file):
            logger.debug("File {0} detected as "
                         "a rar file".format(archive_file))
            arch = rarfile.RarFile(archive_file)

    if not arch:
        raise TypeError("File is not a known archive")

    logger.debug("Extracting files to {0}".format(path))

    try:
        arch.extractall(path=path)
    finally:
        arch.close()

    if delete_on_success:
        logger.debug("Archive {0} will now be deleted".format(archive_file))
        os.unlink(archive_file)

    return os.path.abspath(path)