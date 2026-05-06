def convert_row_to_sdpa_index(block_struct, row_offsets, row):
    """Helper function to map to sparse SDPA index values.
    """
    block_index = bisect_left(row_offsets[1:], row + 1)
    width = block_struct[block_index]
    row = row - row_offsets[block_index]
    i, j = divmod(row, width)
    return block_index, i, j