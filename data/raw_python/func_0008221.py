def create_dir_rec(path: Path):
    """
    Create a folder recursive.

    :param path: path
    :type path: ~pathlib.Path
    """
    if not path.exists():
        Path.mkdir(path, parents=True, exist_ok=True)