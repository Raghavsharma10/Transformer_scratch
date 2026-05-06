def make_box_pixel_mask_from_col_row(column, row, default=0, value=1):
    '''Generate box shaped mask from column and row lists. Takes the minimum and maximum value from each list.

    Parameters
    ----------
    column : iterable, int
        List of colums values.
    row : iterable, int
        List of row values.
    default : int
        Value of pixels that are not selected by the mask.
    value : int
        Value of pixels that are selected by the mask.

    Returns
    -------
    numpy.ndarray
    '''
    # FE columns and rows start from 1
    col_array = np.array(column) - 1
    row_array = np.array(row) - 1
    if np.any(col_array >= 80) or np.any(col_array < 0) or np.any(row_array >= 336) or np.any(row_array < 0):
        raise ValueError('Column and/or row out of range')
    shape = (80, 336)
    mask = np.full(shape, default, dtype=np.uint8)
    if column and row:
        mask[col_array.min():col_array.max() + 1, row_array.min():row_array.max() + 1] = value  # advanced indexing
    return mask