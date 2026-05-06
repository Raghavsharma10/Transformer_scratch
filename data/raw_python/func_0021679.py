def load(cls, filename, gzipped, byteorder='big'):
        """Read, parse and return the file at the specified location.

        The `gzipped` argument is used to indicate if the specified
        file is gzipped. The `byteorder` argument lets you specify
        whether the file is big-endian or little-endian.
        """
        open_file = gzip.open if gzipped else open
        with open_file(filename, 'rb') as buff:
            return cls.from_buffer(buff, byteorder)