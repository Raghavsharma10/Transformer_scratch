def write_moc_ascii(moc, filename=None, file=None):
    """Write a MOC to an ASCII file.

    Either a filename, or an open file object can be specified.
    """

    orders = []

    for (order, cells) in moc:
        ranges = []
        rmin = rmax = None

        for cell in sorted(cells):
            if rmin is None:
                rmin = rmax = cell
            elif rmax == cell - 1:
                rmax = cell
            else:
                ranges.append(_format_range(rmin, rmax))
                rmin = rmax = cell

        ranges.append(_format_range(rmin, rmax))

        orders.append('{0}'.format(order) + '/' + ','.join(ranges))

    if file is not None:
        _write_ascii(orders, file)
    else:
        with open(filename, 'w') as f:
            _write_ascii(orders, f)