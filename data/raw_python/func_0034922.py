def bwrite(stream, obj):
    """ Encode a given object to a file or stream.
    """
    handle = None
    if not hasattr(stream, "write"):
        stream = handle = open(stream, "wb")
    try:
        stream.write(bencode(obj))
    finally:
        if handle:
            handle.close()