def load_experiment(folder, return_path=False):
    '''load_experiment:
    reads in the config.json for a folder, returns None if not found.
    :param folder: full path to experiment folder
    :param return_path: if True, don't load the config.json, but return it
    '''
    fullpath = os.path.abspath(folder)
    config = "%s/config.json" %(fullpath)
    if not os.path.exists(config):
        bot.error("config.json could not be found in %s" %(folder))
        config = None
    if return_path is False and config is not None:
        config = read_json(config)
    return config