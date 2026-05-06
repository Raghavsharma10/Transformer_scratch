def load_user_config(args, log):
    """Load settings from the user's confiuration file, and add them to `args`.

    Settings are loaded from the configuration file in the user's home
    directory.  Those parameters are added (as attributes) to the `args`
    object.

    Arguments
    ---------
    args : `argparse.Namespace`
        Namespace object to which configuration attributes will be added.

    Returns
    -------
    args : `argparse.Namespace`
        Namespace object with added attributes.

    """
    if not os.path.exists(_CONFIG_PATH):
        err_str = (
            "Configuration file does not exists ({}).\n".format(_CONFIG_PATH) +
            "Run `python -m astrocats setup` to configure.")
        log_raise(log, err_str)

    config = json.load(open(_CONFIG_PATH, 'r'))
    setattr(args, _BASE_PATH_KEY, config[_BASE_PATH_KEY])
    log.debug("Loaded configuration: {}: {}".format(_BASE_PATH_KEY, config[
        _BASE_PATH_KEY]))
    return args