def get_command(arguments):
    """Utility function to extract command from docopt arguments.

    :param arguments:
    :return: command
    """
    cmds = list(filter(lambda k: not (k.startswith('-') or
                                 k.startswith('<')) and arguments[k],
                  arguments.keys()))
    if len(cmds) != 1:
        raise Exception('invalid command line!')
    return cmds[0]