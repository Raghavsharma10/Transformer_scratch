def _hash_the_file(hasher, filename):
    """Helper function for creating hash functions.

    See implementation of :func:`dtoolcore.filehasher.shasum`
    for more usage details.
    """
    BUF_SIZE = 65536
    with open(filename, 'rb') as f:
        buf = f.read(BUF_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BUF_SIZE)
    return hasher