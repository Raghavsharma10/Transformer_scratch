def store_config(config, suffix = None):
    '''
    Store configuration

    args:
        config (list[dict]): configurations for each project
    '''
    home = os.path.expanduser('~')
    if suffix is not None:
        config_path = os.path.join(home, '.transfer', suffix)
    else:
        config_path = os.path.join(home, '.transfer')

    os.makedirs(config_path, exist_ok = True)
    with open(os.path.join(config_path, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)