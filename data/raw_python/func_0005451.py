def filtered_walk(path, include=None, exclude=None):
    # type: (str, List[str], List[str]) -> Generator[str]
    """ Walk recursively starting at *path* excluding files matching *exclude*

    Args:
        path (str):
            A starting path. This has to be an existing directory.
        include (list[str]):
            A white list of glob patterns. If given, only files that match those
            globs will be yielded (filtered by exclude).
        exclude (list[str]):
            A list of glob string patterns to test against. If the file/path
            matches any of those patters, it will be filtered out.

    Returns:
        Generator[str]: A generator yielding all the files that do not match any
        pattern in ``exclude``.
    """
    exclude = exclude or []

    if not isdir(path):
        raise ValueError("Cannot walk files, only directories")

    files = os.listdir(path)
    for name in files:
        filename = normpath(join(path, name))

        # If excluded, completely skip it. Will not recurse into directories
        if search_globs(filename, exclude):
            continue

        # If we have a whitelist and the pattern matches, yield it. If the
        # pattern didn't match and it's a dir, it will still be recursively
        # processed.
        if include is None or match_globs(filename, include):
            yield filename

        if isdir(filename):
            for p in filtered_walk(filename, include, exclude):
                yield p