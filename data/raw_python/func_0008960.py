def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        """
        Open the file pointed by this path and return a file object, as
        the built-in open() function does.
        """
        if sys.version_info >= (3, 3):
            return io.open(str(self), mode, buffering, encoding, errors, newline,
                           opener=self._opener)
        else:
            return io.open(str(self), mode, buffering, encoding, errors, newline)