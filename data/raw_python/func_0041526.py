def reapply(config):
    """Reapply current revision"""
    with open(config, 'r'):
        main.reapply(yaml.load(open(config)))