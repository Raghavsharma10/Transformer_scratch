def _last_bookmark(b0, b1):
    """ Return the latest of two bookmarks by looking for the maximum
    integer value following the last colon in the bookmark string.
    """
    n = [None, None]
    _, _, n[0] = b0.rpartition(":")
    _, _, n[1] = b1.rpartition(":")
    for i in range(2):
        try:
            n[i] = int(n[i])
        except ValueError:
            raise ValueError("Invalid bookmark: {}".format(b0))
    return b0 if n[0] > n[1] else b1