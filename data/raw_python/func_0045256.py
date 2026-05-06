def settings(instance):
    """Definition to set settings from config file to the app instance."""
    with open(instance.root_dir + '/Config/config.yml') as config:
        config = yaml.load(config)
        instance.name = config['name']
        instance.port = config['web']['port']
        # default host
        instance.host = "http://localhost"
        if 'host' in config['web']:
            instance.host = config['web']['host']
        instance.debug = config['debug']
    return instance