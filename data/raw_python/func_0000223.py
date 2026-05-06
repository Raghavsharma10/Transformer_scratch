def relative_to(
    source_path: Union[str, pathlib.Path], target_path: Union[str, pathlib.Path]
) -> pathlib.Path:
    """
    Generates relative path from ``source_path`` to ``target_path``.

    Handles the case of paths without a common prefix.

    ::

        >>> import pathlib
        >>> source = pathlib.Path('foo/bar/baz')
        >>> target = pathlib.Path('foo/quux/biz')

    ::

        >>> target.relative_to(source)
        Traceback (most recent call last):
          ...
        ValueError: 'foo/quux/biz' does not start with 'foo/bar/baz'

    ::

        >>> import uqbar.io
        >>> str(uqbar.io.relative_to(source, target))
        '../../quux/biz'

    :param source_path: the source path
    :param target_path: the target path
    """
    source_path = pathlib.Path(source_path).absolute()
    if source_path.is_file():
        source_path = source_path.parent
    target_path = pathlib.Path(target_path).absolute()
    common_prefix = find_common_prefix([source_path, target_path])
    if not common_prefix:
        raise ValueError("No common prefix")
    source_path = source_path.relative_to(common_prefix)
    target_path = target_path.relative_to(common_prefix)
    result = pathlib.Path(*[".."] * len(source_path.parts))
    return result / target_path