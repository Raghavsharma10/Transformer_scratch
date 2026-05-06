def array_bytes(shape, dtype):
    """ Estimates the memory in bytes required for an array of the supplied shape and dtype """
    return np.product(shape)*np.dtype(dtype).itemsize