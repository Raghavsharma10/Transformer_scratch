def show(config):
    """Show revision list"""
    with open(config, 'r'):
        main.show(yaml.load(open(config)))