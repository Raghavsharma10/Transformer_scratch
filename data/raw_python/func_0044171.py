def write_moc_json(moc, filename=None, file=None):
    """Write a MOC in JSON encoding.

    Either a filename, or an open file object can be specified.
    """

    moc.normalize()

    obj = {}

    for (order, cells) in moc:
        obj['{0}'.format(order)] = sorted(cells)

    if file is not None:
        _write_json(obj, file)
    else:
        with open(filename, 'wb') as f:
            _write_json(obj, f)