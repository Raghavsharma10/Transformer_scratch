def get_anchor_diff(anchor):
    """Get the get_anchor_diff between an anchor and the current state of its source.

    Returns: A tuple of get_anchor_diff lines. If there is not different, then this
        returns an empty tuple.
    """
    new_anchor = make_anchor(
        file_path=anchor.file_path,
        offset=anchor.context.offset,
        width=len(anchor.context.topic),
        context_width=anchor.context.width,
        metadata=anchor.metadata)

    assert anchor.file_path == new_anchor.file_path
    assert anchor.context.offset == new_anchor.context.offset
    assert len(anchor.context.topic) == len(new_anchor.context.topic)
    assert anchor.metadata == new_anchor.metadata

    return tuple(
        _context_diff(
            anchor.file_path,
            anchor.context,
            new_anchor.context))