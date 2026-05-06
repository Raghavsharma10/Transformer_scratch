def requirement_args(argv, want_paths=False, want_other=False):
    """Return an iterable of filtered arguments.

    :arg argv: Arguments, starting after the subcommand
    :arg want_paths: If True, the returned iterable includes the paths to any
        requirements files following a ``-r`` or ``--requirement`` option.
    :arg want_other: If True, the returned iterable includes the args that are
        not a requirement-file path or a ``-r`` or ``--requirement`` flag.

    """
    was_r = False
    for arg in argv:
        # Allow for requirements files named "-r", don't freak out if there's a
        # trailing "-r", etc.
        if was_r:
            if want_paths:
                yield arg
            was_r = False
        elif arg in ['-r', '--requirement']:
            was_r = True
        else:
            if want_other:
                yield arg