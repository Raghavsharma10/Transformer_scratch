def _internal_convert(inp):
    """
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    """
    '''
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    '''

    # Read the "magic number" - 4 bytes.
    try:
        mn = struct.unpack('>BBBB', inp.read(4))
    except struct.error:
        raise FormatError(struct.error)

    # First two bytes are always zero, check it.
    if mn[0] != 0 or mn[1] != 0:
        msg = ("Incorrect first two bytes of the magic number: " +
               "0x{0:02X} 0x{1:02X}".format(mn[0], mn[1]))
        raise FormatError(msg)

    # 3rd byte is the data type code.
    dtype_code = mn[2]
    if dtype_code not in _DATA_TYPES_IDX:
        msg = "Incorrect data type code: 0x{0:02X}".format(dtype_code)
        raise FormatError(msg)

    # 4th byte is the number of dimensions.
    dims = int(mn[3])

    # See possible data types description.
    dtype, dtype_s, el_size = _DATA_TYPES_IDX[dtype_code]

    # 4-byte integer for length of each dimension.
    try:
        dims_sizes = struct.unpack('>' + 'I' * dims, inp.read(4 * dims))
    except struct.error as e:
        raise FormatError('Dims sizes: {0}'.format(e))

    # Full length of data.
    full_length = reduce(operator.mul, dims_sizes, 1)

    # Create a numpy array from the data
    try:
        result_array = numpy.frombuffer(
            inp.read(full_length * el_size),
            dtype=numpy.dtype(dtype)
        ).reshape(dims_sizes)
    except ValueError as e:
        raise FormatError('Error creating numpy array: {0}'.format(e))

    # Check for superfluous data.
    if len(inp.read(1)) > 0:
        raise FormatError('Superfluous data detected.')

    return result_array