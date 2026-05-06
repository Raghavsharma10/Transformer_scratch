def sample_dropout_mask(x, dropout_probability=.5, columns=None, stream=None, target=None,
                        dropout_mask=None, dropout_prob_array=None):
    """ Samples a dropout mask and applies it in place"""

    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])

    shape = x.shape

    if dropout_prob_array is None:
        dropout_prob_array = gpuarray.empty(shape, x.dtype, allocator=memory_pool.allocate)
    sampler.fill_uniform(dropout_prob_array, stream)

    if dropout_mask is None:
        dropout_mask = gpuarray.empty(shape, np.int8, allocator=memory_pool.allocate)

    if target is None: target = x
    
    all_kernels['sample_dropout_mask'](
        x, target, dropout_mask, dropout_prob_array,
        np.float32(dropout_probability))

    if columns is not None:
        insert_columns(x, x_tmp, columns[0])

    return dropout_mask