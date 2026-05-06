def get_file_hash(file, seed, bufsz=DEFAULT_BUFFER_SIZE):
        """This method generates a secure hash of the given file.  Returns the
        hash

        :param file: a file like object to get a hash of.  should support
            `read()`
        :param seed: the seed to use for key of the HMAC function
        :param bufsz: an optional buffer size to use for reading the file
        """
        h = hmac.new(seed, None, hashlib.sha256)
        while (True):
            buffer = file.read(bufsz)
            h.update(buffer)
            if (len(buffer) != bufsz):
                break
        return h.digest()