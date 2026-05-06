def cprint(msg, *args, **kw):
    # type: (str, *Any, **Any) -> None
    """ Print colored message to stdout. """
    if len(args) or len(kw):
        msg = msg.format(*args, **kw)

    print(fmt('{}<0>'.format(msg)))