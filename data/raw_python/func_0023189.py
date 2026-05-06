def get(name):
    """Retrieve code from the given filename."""

    filename = find(name)
    if filename is None:
        raise RuntimeError('Could not find %s' % name)
    with open(filename) as fid:
        return fid.read()