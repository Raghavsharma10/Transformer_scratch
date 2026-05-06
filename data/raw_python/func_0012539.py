def file_hash(path, hash_type="md5", block_size=65536, hex_digest=True):
    """
    Hash a given file with md5, or any other and return the hex digest. You
    can run `hashlib.algorithms_available` to see which are available on your
    system unless you have an archaic python version, you poor soul).

    This function is designed to be non memory intensive.

    .. code:: python

        reusables.file_hash(test_structure.zip")
        # '61e387de305201a2c915a4f4277d6663'

    :param path: location of the file to hash
    :param hash_type: string name of the hash to use
    :param block_size: amount of bytes to add to hasher at a time
    :param hex_digest: returned as hexdigest, false will return digest
    :return: file's hash
    """
    hashed = hashlib.new(hash_type)
    with open(path, "rb") as infile:
        buf = infile.read(block_size)
        while len(buf) > 0:
            hashed.update(buf)
            buf = infile.read(block_size)
    return hashed.hexdigest() if hex_digest else hashed.digest()