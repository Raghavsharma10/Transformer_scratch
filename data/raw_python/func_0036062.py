def get_chunk_hash(file,
                       seed,
                       filesz=None,
                       chunksz=DEFAULT_CHUNK_SIZE,
                       bufsz=DEFAULT_BUFFER_SIZE):
        """returns a hash of a chunk of the file provided.  the position of
        the chunk is determined by the seed.  additionally, the hmac of the
        chunk is calculated from the seed.

        :param file: a file like object to get the chunk hash from.  should
        support `read()`, `seek()` and `tell()`.
        :param seed: the seed to use for calculating the chunk position and
        chunk hash
        :param chunksz: the size of the chunk to check
        :param bufsz: an optional buffer size to use for reading the file.
        """
        if (filesz is None):
            file.seek(0, 2)
            filesz = file.tell()
        if (filesz < chunksz):
            chunksz = filesz
        prf = KeyedPRF(seed, filesz - chunksz + 1)
        i = prf.eval(0)
        file.seek(i)
        h = hmac.new(seed, None, hashlib.sha256)
        while (True):
            if (chunksz < bufsz):
                bufsz = chunksz
            buffer = file.read(bufsz)
            h.update(buffer)
            chunksz -= len(buffer)
            assert(chunksz >= 0)
            if (chunksz == 0):
                break
        return h.digest()