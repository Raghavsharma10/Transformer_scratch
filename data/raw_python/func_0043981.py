def _open_repo(args, path_key='<path>'):
    """Open and return the repository containing the specified file.

    The file is specified by looking up `path_key` in `args`. This value or
    `None` is passed to `open_repository`.

    Returns: A `Repository` instance.

    Raises:
        ExitError: If there is a problem opening the repo.
    """
    path = pathlib.Path(args[path_key]) if args[path_key] else None

    try:
        repo = open_repository(path)
    except ValueError as exc:
        raise ExitError(ExitCode.DATA_ERR, str(exc))

    return repo