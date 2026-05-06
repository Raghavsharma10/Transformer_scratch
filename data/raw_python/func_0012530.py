def archive(files_to_archive, name="archive.zip", archive_type=None,
            overwrite=False, store=False, depth=None, err_non_exist=True,
            allow_zip_64=True, **tarfile_kwargs):
    """ Archive a list of files (or files inside a folder), can chose between

        - zip
        - tar
        - gz (tar.gz, tgz)
        - bz2 (tar.bz2)

    .. code:: python

        reusables.archive(['reusables', '.travis.yml'],
                              name="my_archive.bz2")
        # 'C:\\Users\\Me\\Reusables\\my_archive.bz2'

    :param files_to_archive: list of files and folders to archive
    :param name: path and name of archive file
    :param archive_type: auto-detects unless specified
    :param overwrite: overwrite if archive exists
    :param store: zipfile only, True will not compress files
    :param depth: specify max depth for folders
    :param err_non_exist: raise error if provided file does not exist
    :param allow_zip_64: must be enabled for zip files larger than 2GB
    :param tarfile_kwargs: extra args to pass to tarfile.open
    :return: path to created archive
    """
    if not isinstance(files_to_archive, (list, tuple)):
        files_to_archive = [files_to_archive]

    if not archive_type:
        if name.lower().endswith("zip"):
            archive_type = "zip"
        elif name.lower().endswith("gz"):
            archive_type = "gz"
        elif name.lower().endswith("z2"):
            archive_type = "bz2"
        elif name.lower().endswith("tar"):
            archive_type = "tar"
        else:
            err_msg = ("Could not determine archive "
                       "type based off {0}".format(name))
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.debug("{0} file detected for {1}".format(archive_type, name))
    elif archive_type not in ("tar", "gz", "bz2", "zip"):
        err_msg = ("archive_type must be zip, gz, bz2,"
                   " or gz, was {0}".format(archive_type))
        logger.error(err_msg)
        raise ValueError(err_msg)

    if not overwrite and os.path.exists(name):
        err_msg = "File {0} exists and overwrite not specified".format(name)
        logger.error(err_msg)
        raise OSError(err_msg)

    if archive_type == "zip":
        arch = zipfile.ZipFile(name, 'w',
                               zipfile.ZIP_STORED if store else
                               zipfile.ZIP_DEFLATED,
                               allowZip64=allow_zip_64)
        write = arch.write
    elif archive_type in ("tar", "gz", "bz2"):
        mode = archive_type if archive_type != "tar" else ""
        arch = tarfile.open(name, 'w:{0}'.format(mode), **tarfile_kwargs)
        write = arch.add
    else:
        raise ValueError("archive_type must be zip, gz, bz2, or gz")

    try:
        for file_path in files_to_archive:
            if os.path.isfile(file_path):
                if err_non_exist and not os.path.exists(file_path):
                    raise OSError("File {0} does not exist".format(file_path))
                write(file_path)
            elif os.path.isdir(file_path):
                for nf in find_files(file_path, abspath=False, depth=depth):
                    write(nf)
    except (Exception, KeyboardInterrupt) as err:
        logger.exception("Could not archive {0}".format(files_to_archive))
        try:
            arch.close()
        finally:
            os.unlink(name)
        raise err
    else:
        arch.close()

    return os.path.abspath(name)