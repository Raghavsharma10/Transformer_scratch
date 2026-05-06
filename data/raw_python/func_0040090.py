def hash_stream(fileobj, hasher=None, blocksize=65536):
    """Read from fileobj stream, return hash of its contents.

    Args:
      fileobj: File-like object with read()
      hasher: Hash object such as hashlib.sha1(). Defaults to sha1.
      blocksize: Read from fileobj this many bytes at a time.
    """
    hasher = hasher or hashlib.sha1()
    buf = fileobj.read(blocksize)
    while buf:
        hasher.update(buf)
        buf = fileobj.read(blocksize)
    return hasher