def _try_mask_row(row1, row2, all_close, ignore_order):
    '''
    if each value in row1 matches a value in row2, mask row2

    row1
        1d array
    row2
        1d masked array whose mask is all False
    ignore_order : bool
        Ignore column order
    all_close : bool
        compare with np.isclose instead of ==

    Return whether masked the row
    '''
    if ignore_order:
        for value1 in row1:
            if not _try_mask_first_value(value1, row2, all_close):
                row2.mask = ma.nomask
                return False
    else:
        for value1, value2 in zip(row1, row2):
            if not _value_equals(value1, value2, all_close):
                return False
        row2[:] = ma.masked
    assert row2.mask.all()  # sanity check
    return True