def make_pixel_mask(steps, shift, default=0, value=1, enable_columns=None, mask=None):
    '''Generate pixel mask.

    Parameters
    ----------
    steps : int
        Number of mask steps, e.g. steps=3 (every third pixel is enabled), steps=336 (one pixel per column), steps=672 (one pixel per double column).
    shift : int
        Shift mask by given value to the bottom (towards higher row numbers). From 0 to (steps - 1).
    default : int
        Value of pixels that are not selected by the mask.
    value : int
        Value of pixels that are selected by the mask.
    enable_columns : list
        List of columns where the shift mask will be applied. List elements can range from 1 to 80.
    mask : array_like
        Additional mask. Must be convertible to an array of booleans with the same shape as mask array. True indicates a masked (i.e. invalid) data. Masked pixels will be set to default value.

    Returns
    -------
    mask_array : numpy.ndarray
        Mask array.

    Usage
    -----
    shift_mask = 'enable'
    steps = 3 # three step mask
    for mask_step in range(steps):
        commands = []
        commands.extend(self.register.get_commands("ConfMode"))
        mask_array = make_pixel_mask(steps=steps, step=mask_step)
        self.register.set_pixel_register_value(shift_mask, mask_array)
        commands.extend(self.register.get_commands("WrFrontEnd", same_mask_for_all_dc=True, name=shift_mask))
        self.register_utils.send_commands(commands)
        # do something here
    '''
    shape = (80, 336)
    # value = np.zeros(dimension, dtype = np.uint8)
    mask_array = np.full(shape, default, dtype=np.uint8)
    # FE columns and rows are starting from 1
    if enable_columns:
        odd_columns = [odd - 1 for odd in enable_columns if odd % 2 != 0]
        even_columns = [even - 1 for even in enable_columns if even % 2 == 0]
    else:
        odd_columns = range(0, 80, 2)
        even_columns = range(1, 80, 2)
    odd_rows = np.arange(shift % steps, 336, steps)
    even_row_offset = ((steps // 2) + shift) % steps  # // integer devision
    even_rows = np.arange(even_row_offset, 336, steps)
    if odd_columns:
        odd_col_rows = itertools.product(odd_columns, odd_rows)  # get any combination of column and row, no for loop needed
        for odd_col_row in odd_col_rows:
            mask_array[odd_col_row[0], odd_col_row[1]] = value  # advanced indexing
    if even_columns:
        even_col_rows = itertools.product(even_columns, even_rows)
        for even_col_row in even_col_rows:
            mask_array[even_col_row[0], even_col_row[1]] = value
    if mask is not None:
        mask_array = np.ma.array(mask_array, mask=mask, fill_value=default)
        mask_array = mask_array.filled()
    return mask_array