def prepare_dir(app, directory, delete=False):
    """Create apidoc dir, delete contents if delete is True.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param directory: the apidoc directory. you can use relative paths here
    :type directory: str
    :param delete: if True, deletes the contents of apidoc. This acts like an override switch.
    :type delete: bool
    :returns: None
    :rtype: None
    :raises: None
    """
    logger.info("Preparing output directories for jinjaapidoc.")
    if os.path.exists(directory):
        if delete:
            logger.debug("Deleting dir %s", directory)
            shutil.rmtree(directory)
            logger.debug("Creating dir %s", directory)
            os.mkdir(directory)
    else:
        logger.debug("Creating %s", directory)
        os.mkdir(directory)