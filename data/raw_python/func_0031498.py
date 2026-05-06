def to_chunks(stream_or_generator):
    """This generator function receives file-like or generator as input
    and returns generator.

    :param file|__generator[bytes] stream_or_generator: readable stream or
           generator.

    :rtype: __generator[bytes]

    :raise: TypeError
    """

    if isinstance(stream_or_generator, types.GeneratorType):
        yield from stream_or_generator
    elif hasattr(stream_or_generator, 'read'):
        while True:
            chunk = stream_or_generator.read(CHUNK_SIZE)
            if not chunk:
                break  # no more data

            yield chunk

    else:
        raise TypeError('Input must be either readable or generator.')