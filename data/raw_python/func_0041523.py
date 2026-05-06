def current(config):
    """Display current revision"""
    with open(config, 'r'):
        main.current(yaml.load(open(config)))