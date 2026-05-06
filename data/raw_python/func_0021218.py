def watch(logger_name, level=DEBUG, out=stdout):
    """ Quick wrapper for using the Watcher.

    :param logger_name: name of logger to watch
    :param level: minimum log level to show (default INFO)
    :param out: where to send output (default stdout)
    :return: Watcher instance
    """
    watcher = Watcher(logger_name)
    watcher.watch(level, out)
    return watcher