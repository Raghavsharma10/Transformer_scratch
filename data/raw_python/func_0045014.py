def _open_file(self, mode, encoding=None):
        """
        Opens the next current file.

        :param str mode: The mode for opening the file.
        :param str encoding: The encoding of the file.
        """
        if self._filename[-4:] == '.bz2':
            self._file = bz2.open(self._filename, mode=mode, encoding=encoding)
        else:
            self._file = open(self._filename, mode=mode, encoding=encoding)