def _get_sample(self, mode, encoding):
        """
        Get a sample from the next current input file.

        :param str mode: The mode for opening the file.
        :param str|None encoding: The encoding of the file. None for open the file in binary mode.
        """
        self._open_file(mode, encoding)
        self._sample = self._file.read(UniversalCsvReader.sample_size)
        self._file.close()