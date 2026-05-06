def save(self, filename=None, *, gzipped=None, byteorder=None):
        """Write the file at the specified location.

        The `gzipped` keyword only argument indicates if the file should
        be gzipped. The `byteorder` keyword only argument lets you
        specify whether the file should be big-endian or little-endian.

        If the method is called without any argument, it will default to
        the instance attributes and use the file's `filename`,
        `gzipped` and `byteorder` attributes. Calling the method without
        a `filename` will raise a `ValueError` if the `filename` of the
        file is `None`.
        """
        if gzipped is None:
            gzipped = self.gzipped
        if filename is None:
            filename = self.filename

        if filename is None:
            raise ValueError('No filename specified')

        open_file = gzip.open if gzipped else open
        with open_file(filename, 'wb') as buff:
            self.write(buff, byteorder or self.byteorder)