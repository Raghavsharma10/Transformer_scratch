def get_experiments(base, load=False):
    ''' get_experiments will return loaded json for all valid experiments from an experiment folder
    :param base: full path to the base folder with experiments inside
    :param load: if True, returns a list of loaded config.json objects. If False (default) returns the paths to the experiments
    '''
    experiments = find_directories(base)
    valid_experiments = [e for e in experiments if validate(e,cleanup=False)]
    bot.info("Found %s valid experiments" %(len(valid_experiments)))
    if load is True:
        valid_experiments = load_experiments(valid_experiments)

    #TODO at some point in this workflow we would want to grab instructions from help
    # and variables from labels, environment, etc.
    return valid_experiments