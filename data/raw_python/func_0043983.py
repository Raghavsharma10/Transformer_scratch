def init_handler(args):
    """usage: {program} init

    Initialize a new spor repository in the current directory.
    """
    try:
        initialize_repository(pathlib.Path.cwd())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return ExitCode.DATAERR

    return ExitCode.OK