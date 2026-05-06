def icompressed2ibytes(source):
    """
    :param source: GENERATOR OF COMPRESSED BYTES
    :return: GENERATOR OF BYTES
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    last_bytes_count = 0  # Track the last byte count, so we do not show too many debug lines
    bytes_count = 0
    for bytes_ in source:
        try:
            data = decompressor.decompress(bytes_)
        except Exception as e:
            Log.error("problem", cause=e)
        bytes_count += len(data)
        if mo_math.floor(last_bytes_count, 1000000) != mo_math.floor(bytes_count, 1000000):
            last_bytes_count = bytes_count
            DEBUG and Log.note("bytes={{bytes}}", bytes=bytes_count)
        yield data