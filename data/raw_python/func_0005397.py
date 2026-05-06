def _unpack_asf_image(data):
    """Unpack image data from a WM/Picture tag. Return a tuple
    containing the MIME type, the raw image data, a type indicator, and
    the image's description.

    This function is treated as "untrusted" and could throw all manner
    of exceptions (out-of-bounds, etc.). We should clean this up
    sometime so that the failure modes are well-defined.
    """
    type, size = struct.unpack_from('<bi', data)
    pos = 5
    mime = b''
    while data[pos:pos + 2] != b'\x00\x00':
        mime += data[pos:pos + 2]
        pos += 2
    pos += 2
    description = b''
    while data[pos:pos + 2] != b'\x00\x00':
        description += data[pos:pos + 2]
        pos += 2
    pos += 2
    image_data = data[pos:pos + size]
    return (mime.decode("utf-16-le"), image_data, type,
            description.decode("utf-16-le"))