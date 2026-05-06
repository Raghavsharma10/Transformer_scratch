def load_experiments(folders):
    '''load_experiments
    a wrapper for load_experiment to read multiple experiments
    :param experiment_folders: a list of experiment folders to load, full paths
    '''
    experiments = []
    if isinstance(folders,str):
        folders = [experiment_folders]
    for folder in folders:
        exp = load_experiment(folder)
        experiments.append(exp)
    return experiments