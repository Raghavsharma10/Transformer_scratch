def deepcopy_strip(item):  # type: (Any) -> Any
    """
    Make a deep copy of list and dict objects.

    Intentionally do not copy attributes.  This is to discard CommentedMap and
    CommentedSeq metadata which is very expensive with regular copy.deepcopy.
    """

    if isinstance(item, MutableMapping):
        return {k: deepcopy_strip(v) for k, v in iteritems(item)}
    if isinstance(item, MutableSequence):
        return [deepcopy_strip(k) for k in item]
    return item