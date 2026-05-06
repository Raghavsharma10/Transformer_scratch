def _try_mask_first_value(value, row, all_close):
    '''
    mask first value in row

    value1 : ~typing.Any
    row : 1d masked array
    all_close : bool
        compare with np.isclose instead of ==

    Return whether masked a value
    '''
    # Compare value to row
    for i, value2 in enumerate(row):
        if _value_equals(value, value2, all_close):
            row[i] = ma.masked
            return True
    return False