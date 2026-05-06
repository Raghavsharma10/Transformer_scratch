def read_from_bpch(filename, file_position, shape, dtype, endian,
                   use_mmap=False):
    """ Read a chunk of data from a bpch output file.

    Parameters
    ----------
    filename : str
        Path to file on disk containing the  data
    file_position : int
        Position (bytes) where desired data chunk begins
    shape : tuple of ints
        Resultant (n-dimensional) shape of requested data; the chunk
        will be read sequentially from disk and then re-shaped
    dtype : dtype
        Dtype of data; for best results, pass a dtype which includes
        an endian indicator, e.g. `dtype = np.dtype('>f4')`
    endian : str
        Endianness of data; should be consistent with `dtype`
    use_mmap : bool
        Memory map the chunk of data to the file on disk, else read
        immediately

    Returns
    -------
    Array with shape `shape` and dtype `dtype` containing the requested
    chunk of data from `filename`.

    """
    offset = file_position + 4
    if use_mmap:
        d = np.memmap(filename, dtype=dtype, mode='r', shape=shape,
                      offset=offset, order='F')
    else:
        with FortranFile(filename, 'rb', endian) as ff:
            ff.seek(file_position)
            d = np.array(ff.readline('*f'))
            d = d.reshape(shape, order='F')

    # As a sanity check, *be sure* that the resulting data block has the
    # correct shape, and fail early if it doesn't.
    if (d.shape != shape):
        raise IOError("Data chunk read from {} does not have the right shape,"
                      " (expected {} but got {})"
                      .format(filename, shape, d.shape))

    return d