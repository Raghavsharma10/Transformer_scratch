def _find_root_dir(path, spor_dir):
    """Search for a spor repo containing `path`.

    This searches for `spor_dir` in directories dominating `path`. If a
    directory containing `spor_dir` is found, then that directory is returned
    as a `pathlib.Path`.

    Returns: The dominating directory containing `spor_dir` as a
      `pathlib.Path`.

    Raises:
      ValueError: No repository is found.

    """

    start_path = pathlib.Path(os.getcwd() if path is None else path)
    paths = [start_path] + list(start_path.parents)

    for path in paths:
        data_dir = path / spor_dir
        if data_dir.exists() and data_dir.is_dir():
            return path

    raise ValueError('No spor repository found')