def validate(folder=None, cleanup=False):
    '''validate
    :param folder: full path to experiment folder with config.json. If path begins
                   with https, we assume to be starting from a repository.
    '''
    from expfactory.validator import ExperimentValidator
    cli = ExperimentValidator()
    return cli.validate(folder, cleanup=cleanup)