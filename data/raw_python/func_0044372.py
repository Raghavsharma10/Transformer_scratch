def make_anchor(file_path: pathlib.Path,
                offset: int,
                width: int,
                context_width: int,
                metadata,
                encoding: str = 'utf-8',
                handle=None):
    """Construct a new `Anchor`.

    Args:
        file_path: The absolute path to the target file for the anchor.
        offset: The offset of the anchored text in codepoints in `file_path`'s
            contents.
        width: The width in codepoints of the anchored text.
        context_width: The width in codepoints of context on either side of the
            anchor.
        metadata: The metadata to attach to the anchor. Must be json-serializeable.
        encoding: The encoding of the contents of `file_path`.
        handle: If not `None`, this is a file-like object the contents of which
            are used to calculate the context of the anchor. If `None`, then
            the file indicated by `file_path` is opened instead.

    Raises:
        ValueError: `width` characters can't be read at `offset`.
        ValueError: `file_path` is not absolute.

    """

    @contextmanager
    def get_handle():
        if handle is None:
            with file_path.open(mode='rt', encoding=encoding) as fp:
                yield fp
        else:
            yield handle

    with get_handle() as fp:
        context = _make_context(fp, offset, width, context_width)

    return Anchor(
        file_path=file_path,
        encoding=encoding,
        context=context,
        metadata=metadata)