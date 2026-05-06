def walk(
    root_path: Union[str, pathlib.Path], top_down: bool = True
) -> Generator[
    Tuple[pathlib.Path, Sequence[pathlib.Path], Sequence[pathlib.Path]], None, None
]:
    """
    Walks a directory tree.

    Like :py:func:`os.walk` but yielding instances of :py:class:`pathlib.Path`
    instead of strings.

    :param root_path: foo
    :param top_down: bar
    """
    root_path = pathlib.Path(root_path)
    directory_paths, file_paths = [], []
    for path in sorted(root_path.iterdir()):
        if path.is_dir():
            directory_paths.append(path)
        else:
            file_paths.append(path)
    if top_down:
        yield root_path, directory_paths, file_paths
    for directory_path in directory_paths:
        yield from walk(directory_path, top_down=top_down)
    if not top_down:
        yield root_path, directory_paths, file_paths