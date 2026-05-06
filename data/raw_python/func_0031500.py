def read_body_stream(stream, chunked=False, compression=None):
    """Read HTTP body stream, yielding blocks of bytes. De-chunk and
    de-compress data if needed.

    :param file stream: readable stream.
    :param bool chunked: whether stream is chunked.
    :param str|None compression: compression type is stream is
           compressed, otherwise None.

    :rtype: __generator[bytes]
    :raise: TypeError, BodyStreamError
    """

    if not (chunked or compression):
        return to_chunks(stream)

    generator = stream
    if chunked:
        generator = dechunk(generator)

    if compression:
        generator = decompress(to_chunks(generator), compression)

    return generator