def convert_to_mosek_index(block_struct, row_offsets, block_offsets, row):
    """MOSEK requires a specific sparse format to define the lower-triangular
    part of a symmetric matrix. This function does the conversion from the
    sparse upper triangular matrix format of Ncpol2SDPA.
    """
    block_index, i, j = convert_row_to_sdpa_index(block_struct, row_offsets,
                                                  row)

    offset = block_offsets[block_index]
    ci = offset + i
    cj = offset + j
    return cj, ci