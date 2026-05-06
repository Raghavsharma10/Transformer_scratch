def set_logfile(path, instance):
    """Specify logfile path"""

    global logfile
    logfile = os.path.normpath(path) + '/hfos.' + instance + '.log'