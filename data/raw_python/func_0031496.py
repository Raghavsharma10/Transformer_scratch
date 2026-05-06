def read_until(stream, delimiter, max_bytes=16):
    """Read until we have found the given delimiter.

    :param file stream: readable file-like object.
    :param bytes delimiter: delimiter string.
    :param int max_bytes: maximum bytes to read.

    :rtype: bytes|None
    """

    buf = bytearray()
    delim_len = len(delimiter)

    while len(buf) < max_bytes:
        c = stream.read(1)

        if not c:
            break

        buf += c
        if buf[-delim_len:] == delimiter:
            return bytes(buf[:-delim_len])