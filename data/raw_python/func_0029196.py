def get_command(arguments):
    """Extract the first argument from arguments parsed by docopt.

    :param arguments parsed by docopt:
    :return: command
    """
    return [k for k, v in arguments.items()
            if not k.startswith('-') and v is True][0]