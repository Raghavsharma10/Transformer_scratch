def _decode_doubles(message):
    """Helper for decode_qp, decodes a double array.

    The double array is stored as little endian 64 bit doubles.
    The array has then been base64 encoded. Since we are decoding we do these
    steps in reverse.

    Args:
        message: the double array

    Returns:
        decoded double array
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('d' * (len(binary) // 8)), binary)