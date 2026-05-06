def iter_all_children(obj, skipContainers=False):
    """
    Returns an iterator over all childen and nested children using
    obj's get_children() method

    if skipContainers is true, only childless objects are returned.
    """
    if hasattr(obj, 'get_children') and len(obj.get_children()) > 0:
        for child in obj.get_children():
            if not skipContainers:
                yield child
            # could use `yield from` in python 3...
            for grandchild in iter_all_children(child, skipContainers):
                yield grandchild
    else:
        yield obj