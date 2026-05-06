def find_common_prefix(
    paths: Sequence[Union[str, pathlib.Path]]
) -> Optional[pathlib.Path]:
    """
    Find the common prefix of two or more paths.

    ::

        >>> import pathlib
        >>> one = pathlib.Path('foo/bar/baz')
        >>> two = pathlib.Path('foo/quux/biz')
        >>> three = pathlib.Path('foo/quux/wuux')

    ::

        >>> import uqbar.io
        >>> str(uqbar.io.find_common_prefix([one, two, three]))
        'foo'

    :param paths: paths to inspect
    """
    counter: collections.Counter = collections.Counter()
    for path in paths:
        path = pathlib.Path(path)
        counter.update([path])
        counter.update(path.parents)
    valid_paths = sorted(
        [path for path, count in counter.items() if count >= len(paths)],
        key=lambda x: len(x.parts),
    )
    if valid_paths:
        return valid_paths[-1]
    return None