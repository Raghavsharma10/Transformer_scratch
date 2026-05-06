def cp(src, dst, overwrite=False):
    """
    Copy files to a new location.

    :param src: list (or string) of paths of files to copy
    :param dst: file or folder to copy item(s) to
    :param overwrite: IF the file already exists, should I overwrite it?
    """

    if not isinstance(src, list):
        src = [src]

    dst = os.path.expanduser(dst)
    dst_folder = os.path.isdir(dst)

    if len(src) > 1 and not dst_folder:
        raise OSError("Cannot copy multiple item to same file")

    for item in src:
        source = os.path.expanduser(item)
        destination = (dst if not dst_folder else
                       os.path.join(dst, os.path.basename(source)))
        if not overwrite and os.path.exists(destination):
            _logger.warning("Not replacing {0} with {1}, overwrite not enabled"
                            "".format(destination, source))
            continue

        shutil.copy(source, destination)