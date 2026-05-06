def _internal_write(out_stream, arr):
    """
    Writes numpy.ndarray arr to a file-like object (with write() method) in
    IDX format.
    """

    if arr.size == 0:
        raise FormatError('Cannot encode empty array.')

    try:
        type_byte, struct_lib_type = _DATA_TYPES_NUMPY[str(arr.dtype)]
    except KeyError:
        raise FormatError('numpy ndarray type not supported by IDX format.')

    if arr.ndim > _MAX_IDX_DIMENSIONS:
        raise FormatError(
            'IDX format cannot encode array with dimensions > 255')

    if max(arr.shape) > _MAX_AXIS_LENGTH:
        raise FormatError('IDX format cannot encode array with more than ' +
                          str(_MAX_AXIS_LENGTH) + ' elements along any axis')

    # Write magic number
    out_stream.write(struct.pack('BBBB', 0, 0, type_byte, arr.ndim))

    # Write array dimensions
    out_stream.write(struct.pack('>' + 'I' * arr.ndim, *arr.shape))

    # Horrible hack to deal with horrible bug when using struct.pack to encode
    # unsigned ints in 2.7 and lower, see http://bugs.python.org/issue2263
    if sys.version_info < (2, 7) and str(arr.dtype) == 'uint8':
        arr_as_list = [int(i) for i in arr.reshape(-1)]
        out_stream.write(struct.pack('>' + struct_lib_type * arr.size,
                                     *arr_as_list))
    else:
        # Write array contents - note that the limit to number of arguments
        # doesn't apply to unrolled arguments
        out_stream.write(struct.pack('>' + struct_lib_type * arr.size,
                                     *arr.reshape(-1)))