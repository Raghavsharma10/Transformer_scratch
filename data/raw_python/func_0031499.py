def decompress(chunks, compression):
    """Decompress

    :param __generator[bytes] chunks: compressed body chunks.
    :param str compression: compression constant.

    :rtype: __generator[bytes]
    :return: decompressed chunks.

    :raise: TypeError, DecompressError
    """

    if compression not in SUPPORTED_COMPRESSIONS:
        raise TypeError('Unsupported compression type: %s' % (compression,))

    de_compressor = DECOMPRESSOR_FACTORIES[compression]()
    try:
        for chunk in chunks:
            try:
                yield de_compressor.decompress(chunk)
            except OSError as err:
                # BZ2Decompressor: invalid data stream
                raise DecompressError(err) from None

        # BZ2Decompressor does not support flush() method.
        if hasattr(de_compressor, 'flush'):
            yield de_compressor.flush()

    except zlib.error as err:
        raise DecompressError(err) from None