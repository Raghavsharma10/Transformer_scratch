def revision(config, message):
    """Create new revision file in a scripts directory"""
    with open(config, 'r'):
        main.revision(yaml.load(open(config)), message)