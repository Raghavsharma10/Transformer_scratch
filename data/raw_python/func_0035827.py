def _try_mask_first_row(row, values, all_close, ignore_order):
    '''
    mask first row in 2d array

    values : 2d masked array
        Each row is either fully masked or not masked at all
    ignore_order : bool
        Ignore column order

    Return whether masked a row. If False, masked nothing.
    '''
    for row2 in values:
        mask = ma.getmaskarray(row2)
        assert mask.sum() in (0, len(mask))  # sanity check: all or none masked
        if mask[0]: # Note: at this point row2's mask is either all False or all True
            continue

        # mask each value of row1 in row2
        if _try_mask_row(row, row2, all_close, ignore_order):
            return True
    # row did not match
    return False