def safe_evaluate(command, glob, local):
    """
    Continue to attempt to execute the given command, importing objects which
    cause a NameError in the command

    :param command: command for eval
    :param glob: globals dict for eval
    :param local: locals dict for eval
    :return: command result
    """
    while True:
        try:
            return eval(command, glob, local)
        except NameError as e:
            match = re.match("name '(.*)' is not defined", e.message)
            if not match:
                raise e
            try:
                exec ('import %s' % (match.group(1), )) in glob
            except ImportError:
                raise e