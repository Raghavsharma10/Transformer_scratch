def read(self, filename, encoding=None):
        """Read and parse a filename.

        Args:
            filename (str): path to file
            encoding (str): encoding of file, default None
        """
        with open(filename, encoding=encoding) as fp:
            self._read(fp, filename)
        self._filename = os.path.abspath(filename)