def load_config_module():
    """
    If the config.py file exists, import it as a module. If it does not exist,
    call sys.exit() with a request to run oaepub configure.
    """
    import imp
    config_path = config_location()
    try:
        config = imp.load_source('config', config_path)
    except IOError:
        log.critical('Config file not found. oaepub exiting...')
        sys.exit('Config file not found. Please run \'oaepub configure\'')
    else:
        log.debug('Config file loaded from {0}'.format(config_path))
        return config