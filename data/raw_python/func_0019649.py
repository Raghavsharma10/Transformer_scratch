def get_args(parser):
    """
    Converts arguments extracted from a parser to a dict,
    and will dismiss arguments which default to NOT_SET.

    :param parser: an ``argparse.ArgumentParser`` instance.
    :type parser: argparse.ArgumentParser
    :return: Dictionary with the configs found in the parsed CLI arguments.
    :rtype: dict
    """
    args = vars(parser.parse_args()).items()
    return {key: val for key, val in args if not isinstance(val, NotSet)}