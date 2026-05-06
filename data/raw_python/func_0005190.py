def err(msg, *args, **kw):
    # type: (str, *Any, **Any) -> None
    """ Per step status messages

    Use this locally in a command definition to highlight more important
    information.
    """
    if len(args) or len(kw):
        msg = msg.format(*args, **kw)

    shell.cprint('-- <31>{}<0>'.format(msg))