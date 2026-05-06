def read_moc_fits_hdu(moc, hdu, include_meta=False):
    """Read data from a FITS table HDU into a MOC.
    """

    if include_meta:
        header = hdu.header

        if 'MOCTYPE' in header:
            moc.type = header['MOCTYPE']
        if 'MOCID' in header:
            moc.id = header['MOCID']
        if 'ORIGIN' in header:
            moc.origin = header['ORIGIN']
        if 'EXTNAME' in header:
            moc.name = header['EXTNAME']

    current_order = None
    current_cells = []

    # Determine type to use for orders: 32 bit if column type is J,
    # otherwise assume we need 64 bit.
    moc_type = np.int32 if (hdu.data.formats[0] == 'J') else np.int64

    nuniqs = hdu.data.field(0)
    orders = (np.log2(nuniqs / 4) / 2).astype(moc_type)
    cells = nuniqs - 4 * (4 ** orders)

    for (order, cell) in izip(orders, cells):
        if order != current_order:
            if current_cells:
                moc.add(current_order, current_cells)

            current_order = order
            current_cells = [cell]

        else:
            current_cells.append(cell)

    if current_cells:
        moc.add(current_order, current_cells)