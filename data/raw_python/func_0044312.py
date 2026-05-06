def initialize_repository(path, spor_dir='.spor'):
    """Initialize a spor repository in `path` if one doesn't already exist.

    Args:
        path: Path to any file or directory within the repository.
        spor_dir: The name of the directory containing spor data.

    Returns: A `Repository` instance.

    Raises:
        ValueError: A repository already exists at `path`.
    """
    path = pathlib.Path(path)
    spor_path = path / spor_dir
    if spor_path.exists():
        raise ValueError('spor directory already exists: {}'.format(spor_path))
    spor_path.mkdir()

    return Repository(path, spor_dir)