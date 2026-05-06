def add_vec_to_mat(mat, vec, axis=None, inplace=False,
                   target=None, substract=False):
    """ Add a vector to a matrix
    """

    assert mat.flags.c_contiguous

    if axis is None:
        if vec.shape[0] == mat.shape[0]:
            axis = 0
        elif vec.shape[0] == mat.shape[1]:
            axis = 1
        else:
            raise ValueError('Vector length must be equal '
                             'to one side of the matrix')

    n, m = mat.shape

    block = (_compilation_constants['add_vec_block_size'],
             _compilation_constants['add_vec_block_size'], 1)
    gridx = ceil_div(n, block[0])
    gridy = ceil_div(m, block[1])
    grid = (gridx, gridy, 1)

    if inplace:
        target = mat
    elif target is None:
        target = gpuarray.empty_like(mat)

    if axis == 0:
        assert vec.shape[0] == mat.shape[0]
        add_col_vec_kernel.prepared_call(
            grid, block,
            mat.gpudata,
            vec.gpudata,
            target.gpudata,
            np.uint32(n),
            np.uint32(m),
            np.int32(substract))
    elif axis == 1:
        assert vec.shape[0] == mat.shape[1]
        add_row_vec_kernel.prepared_call(
            grid, block,
            mat.gpudata,
            vec.gpudata,
            target.gpudata,
            np.uint32(n),
            np.uint32(m),
            np.int32(substract))
    return target