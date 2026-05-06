def bread(stream):
    """ Decode a file or stream to an object.
    """
    if hasattr(stream, "read"):
        return bdecode(stream.read())
    else:
        handle = open(stream, "rb")
        try:
            return bdecode(handle.read())
        finally:
            handle.close()