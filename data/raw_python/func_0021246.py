def last_bookmark(bookmarks):
    """ The bookmark returned by the last :class:`.Transaction`.
    """
    last = None
    for bookmark in bookmarks:
        if last is None:
            last = bookmark
        else:
            last = _last_bookmark(last, bookmark)
    return last