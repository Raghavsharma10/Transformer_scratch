def get(args):
    """Get an Aegea configuration parameter by name"""
    from . import config
    for key in args.key.split("."):
        config = getattr(config, key)
    print(json.dumps(config))