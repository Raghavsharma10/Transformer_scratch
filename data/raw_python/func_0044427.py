def is_inside(directory, fname):
    """True if fname is inside directory.

    The parameters should typically be passed to osutils.normpath first, so
    that . and .. and repeated slashes are eliminated, and the separators
    are canonical for the platform.

    The empty string as a dir name is taken as top-of-tree and matches
    everything.
    """
    # XXX: Most callers of this can actually do something smarter by
    # looking at the inventory
    if directory == fname:
        return True

    if directory == b'':
        return True

    if not directory.endswith(b'/'):
        directory += b'/'

    return fname.startswith(directory)