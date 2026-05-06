def get_options(config_options, local_options, cli_options):
    """
    Figure out what options to use based on the four places it can come from.

    Order of precedence:
    * cli_options      specified by the user at the command line
    * local_options    specified in the config file for the metric
    * config_options   specified in the config file at the base
    * DEFAULT_OPTIONS  hard coded defaults
    """
    options = DEFAULT_OPTIONS.copy()
    if config_options is not None:
        options.update(config_options)
    if local_options is not None:
        options.update(local_options)
    if cli_options is not None:
        options.update(cli_options)
    return options