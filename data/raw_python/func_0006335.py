def make_xtalk_mask(mask):
    """
    Generate xtalk mask (row - 1, row + 1) from pixel mask.

    Parameters
    ----------
    mask : ndarray
        Pixel mask.

    Returns
    -------
    ndarray
        Xtalk mask.

    Example
    -------
    Input:
    [[1 0 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 1 ... 0 1 0 0 0 0 0 1 0 0]
     ...
     [1 0 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 1 ... 0 1 0 0 0 0 0 1 0 0]]

    Output:
    [[0 1 0 0 0 1 0 1 0 0 ... 0 0 0 1 0 1 0 0 0 1]
     [0 0 1 0 1 0 0 0 1 0 ... 1 0 1 0 0 0 1 0 1 0]
     ...
     [0 1 0 0 0 1 0 1 0 0 ... 0 0 0 1 0 1 0 0 0 1]
     [0 0 1 0 1 0 0 0 1 0 ... 1 0 1 0 0 0 1 0 1 0]]
    """
    col, row = mask.nonzero()
    row_plus_one = row + 1
    del_index = np.where(row_plus_one > 335)
    row_plus_one = np.delete(row_plus_one, del_index)
    col_plus_one = np.delete(col.copy(), del_index)
    row_minus_one = row - 1
    del_index = np.where(row_minus_one > 335)
    row_minus_one = np.delete(row_minus_one, del_index)
    col_minus_one = np.delete(col.copy(), del_index)
    col = np.concatenate((col_plus_one, col_minus_one))
    row = np.concatenate((row_plus_one, row_minus_one))
    return make_pixel_mask_from_col_row(col + 1, row + 1)