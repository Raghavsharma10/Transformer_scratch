def dechunk(stream):
    """De-chunk HTTP body stream.

    :param file stream: readable file-like object.

    :rtype: __generator[bytes]
    :raise: DechunkError
    """

    # TODO(vovan): Add support for chunk extensions:
    # TODO(vovan): http://tools.ietf.org/html/rfc2616#section-3.6.1

    while True:
        chunk_len = read_until(stream, b'\r\n')

        if chunk_len is None:
            raise DechunkError(
                'Could not extract chunk size: unexpected end of data.')

        try:
            chunk_len = int(chunk_len.strip(), 16)
        except (ValueError, TypeError) as err:
            raise DechunkError('Could not parse chunk size: %s' % (err,))

        if chunk_len == 0:
            break

        bytes_to_read = chunk_len
        while bytes_to_read:
            chunk = stream.read(bytes_to_read)
            bytes_to_read -= len(chunk)
            yield chunk

        # chunk ends with \r\n
        crlf = stream.read(2)
        if crlf != b'\r\n':
            raise DechunkError('No CR+LF at the end of chunk!')