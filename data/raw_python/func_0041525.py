def checkout(config, rev):
    """Upgrade/revert to a different revision.
    
    <rev> must be "head", integer or revision id. To pass negative
    number you need to write "--" before it"""
    with open(config, 'r'):
        main.checkout(yaml.load(open(config)), rev)