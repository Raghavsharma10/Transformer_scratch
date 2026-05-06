def set_data_dir(directory=None, create=False, save=False):
    """Set vispy data download directory

    Parameters
    ----------
    directory : str | None
        The directory to use.
    create : bool
        If True, create directory if it doesn't exist.
    save : bool
        If True, save the configuration to the vispy config.
    """
    if directory is None:
        directory = _data_path
        if _data_path is None:
            raise IOError('default path cannot be determined, please '
                          'set it manually (directory != None)')
    if not op.isdir(directory):
        if not create:
            raise IOError('directory "%s" does not exist, perhaps try '
                          'create=True to create it?' % directory)
        os.mkdir(directory)
    config.update(data_path=directory)
    if save:
        save_config(data_path=directory)