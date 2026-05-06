def tag(iterable, tags=None, key='@tags'):
    """
    Add tags to each dict or dict-like object in ``iterable``. Tags are added
    to each dict with a key set by ``key``. If a key already exists under the
    key given by ``key``, this function will attempt to ``.extend()``` it, but
    will fall back to replacing it in the event of error.
    """
    if not tags:
        for item in iterable:
            yield item

    else:
        for item in iterable:
            yield _tag(item, tags, key)