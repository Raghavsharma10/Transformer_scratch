def path_glob(pattern, current_dir=None):
    """Use pathlib for ant-like patterns, like: "**/*.py"

    :param pattern:      File/directory pattern to use (as string).
    :param current_dir:  Current working directory (as Path, pathlib.Path, str)
    :return Resolved Path (as path.Path).
    """
    if not current_dir:
        current_dir = pathlib.Path.cwd()
    elif not isinstance(current_dir, pathlib.Path):
        # -- CASE: string, path.Path (string-like)
        current_dir = pathlib.Path(str(current_dir))

    for p in current_dir.glob(pattern):
        yield Path(str(p))