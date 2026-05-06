def info(msg, *args, **kw):
    # type: (str, *Any, **Any) -> None
    """ Print sys message to stdout.

    System messages should inform about the flow of the script. This should
    be a major milestones during the build.
    """
    if len(args) or len(kw):
        msg = msg.format(*args, **kw)

    shell.cprint('-- <32>{}<0>'.format(msg))