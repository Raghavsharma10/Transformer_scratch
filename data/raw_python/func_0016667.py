def _decode_ints(message):
    """Helper for decode_qp, decodes an int array.

    The int array is stored as little endian 32 bit integers.
    The array has then been base64 encoded. Since we are decoding we do these
    steps in reverse.
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('i' * (len(binary) // 4)), binary)