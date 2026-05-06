def _get_parser(f):
    """
    Gets the parser for the command f, if it not exists it creates a new one
    """
    _COMMAND_GROUPS[f.__module__].load()

    if f.__name__ not in _COMMAND_GROUPS[f.__module__].parsers:
        parser = _COMMAND_GROUPS[f.__module__].parser_generator.add_parser(f.__name__, help=f.__doc__,
                                                                           description=f.__doc__)
        parser.set_defaults(func=f)

        _COMMAND_GROUPS[f.__module__].parsers[f.__name__] = parser

    return _COMMAND_GROUPS[f.__module__].parsers[f.__name__]