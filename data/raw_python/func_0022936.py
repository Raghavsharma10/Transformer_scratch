def _extract_options(config, options, *args):
    """Extract options values from a configparser, optparse pair.

    Options given on command line take precedence over options read in the
    configuration file.

    Args:
        config (dict): option values read from a config file through
            configparser
        options (optparse.Options): optparse 'options' object containing options
            values from the command line
        *args (str tuple): name of the options to extract
    """
    extract = {}
    for key in args:
        if key not in args:
            continue
        extract[key] = config[key]
        option = getattr(options, key, None)
        if option is not None:
            extract[key] = option
    return extract