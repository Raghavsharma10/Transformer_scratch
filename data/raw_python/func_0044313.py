def open_repository(path, spor_dir='.spor'):
    """Open an existing repository.

    Args:
        path: Path to any file or directory within the repository.
        spor_dir: The name of the directory containing spor data.

    Returns: A `Repository` instance.

    Raises:
        ValueError: No repository is found.
    """
    root = _find_root_dir(path, spor_dir)
    return Repository(root, spor_dir)