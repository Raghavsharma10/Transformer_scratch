def get_path_list(opts):
    """
    Return a list of all root paths where JS files can be found, given the
    command line options (in dict form) for this script.
    """
    paths = []
    for opt, arg in list(opts.items()):
        if opt in ('-p', '--jspath'):
            paths.append(arg)
    return paths or [os.getcwd()]