def generate_configfile_names(config_files, config_searchpath=None):
    """Generates all configuration file name combinations to read.

    .. sourcecode::

        # -- ALGORITHM:
        #    First basenames/directories are prefered and override other files.
        for config_path in reversed(config_searchpath):
            for config_basename in reversed(config_files):
                config_fname = os.path.join(config_path, config_basename)
                if os.path.isfile(config_fname):
                    yield config_fname

    :param config_files:        List of config file basenames.
    :param config_searchpath:   List of directories to look for config files.
    :return: List of available configuration file names (as generator)
    """
    if config_searchpath is None:
        config_searchpath = ["."]

    for config_path in reversed(config_searchpath):
        for config_basename in reversed(config_files):
            config_fname = os.path.join(config_path, config_basename)
            if os.path.isfile(config_fname):
                # MAYBE: yield os.path.normpath(config_fname)
                yield config_fname