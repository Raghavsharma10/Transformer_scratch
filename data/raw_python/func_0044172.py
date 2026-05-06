def read_moc_json(moc, filename=None, file=None):
    """Read JSON encoded data into a MOC.

    Either a filename, or an open file object can be specified.
    """

    if file is not None:
        obj = _read_json(file)
    else:
        with open(filename, 'rb') as f:
            obj = _read_json(f)

    for (order, cells) in obj.items():
        moc.add(order, cells)