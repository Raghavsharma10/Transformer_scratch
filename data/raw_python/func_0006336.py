def make_checkerboard_mask(column_distance, row_distance, column_offset=0, row_offset=0, default=0, value=1):
    """
    Generate chessboard/checkerboard mask.

    Parameters
    ----------
    column_distance : int
        Column distance of the enabled pixels.
    row_distance : int
        Row distance of the enabled pixels.
    column_offset : int
        Additional column offset which shifts the columns by the given amount.
    column_offset : int
        Additional row offset which shifts the rows by the given amount.

    Returns
    -------
    ndarray
        Chessboard mask.

    Example
    -------
    Input:
    column_distance : 6
    row_distance : 2

    Output:
    [[1 0 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 1 ... 0 1 0 0 0 0 0 1 0 0]
     ...
     [0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 1 ... 0 1 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0]]
    """
    col_shape = (336,)
    col = np.full(col_shape, fill_value=default, dtype=np.uint8)
    col[::row_distance] = value
    shape = (80, 336)
    chessboard_mask = np.full(shape, fill_value=default, dtype=np.uint8)
    chessboard_mask[column_offset::column_distance * 2] = np.roll(col, row_offset)
    chessboard_mask[column_distance + column_offset::column_distance * 2] = np.roll(col, row_distance / 2 + row_offset)
    return chessboard_mask