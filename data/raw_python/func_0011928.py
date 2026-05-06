def includeme(config):
    """Pyramid pluggable and discoverable function."""
    global_settings = config.registry.settings
    settings = local_settings(global_settings, PREFIX)

    try:
        file = settings['file']
    except KeyError:
        raise KeyError("Must supply '{}.file' configuration value "
                       "in order to configure logging via '{}'."
                       .format(PREFIX, PROJECT))

    with open(file, 'r') as f:
        logging_config = yaml.load(f)

    dictConfig(logging_config)

    # Enable transit logging?
    if asbool(settings.get('transit_logging.enabled?', False)):
        config.add_tween('pyramid_sawing.main.TransitLogger')