def base(context, config, database, root, log_level):
    """Housekeeper - Access your files!"""
    coloredlogs.install(level=log_level)
    context.obj = ruamel.yaml.safe_load(config) if config else {}
    context.obj['database'] = database if database else context.obj['database']
    context.obj['root'] = root if root else context.obj['root']