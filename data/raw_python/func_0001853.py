def args_to_dict(args):
    # type: (str) -> DictUpperBound[str,str]
    """Convert command line arguments in a comma separated string to a dictionary

    Args:
        args (str): Command line arguments

    Returns:
        DictUpperBound[str,str]: Dictionary of arguments

    """
    arguments = dict()
    for arg in args.split(','):
        key, value = arg.split('=')
        arguments[key] = value
    return arguments