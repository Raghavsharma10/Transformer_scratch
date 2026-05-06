def from_buffer(cls, buff, byteorder='big'):
        """Load nbt file from a file-like object.

        The `buff` argument can be either a standard `io.BufferedReader`
        for uncompressed nbt or a `gzip.GzipFile` for gzipped nbt data.
        """
        self = cls.parse(buff, byteorder)
        self.filename = getattr(buff, 'name', self.filename)
        self.gzipped = isinstance(buff, gzip.GzipFile)
        self.byteorder = byteorder
        return self