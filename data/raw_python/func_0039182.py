def makedirs(path):
    """Creates the directory tree if non existing."""
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)