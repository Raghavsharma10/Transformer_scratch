def write(self, filehandle, fileformat):
        """Write :class:`~nmrstarlib.plsimulator.PeakList` data into file.

        :param filehandle: file-like object.
        :type filehandle: :py:class:`io.TextIOWrapper`
        :param str fileformat: Format to use to write data: `sparky`, `autoassign`, or `json`.
        :return: None
        :rtype: :py:obj:`None`
        """
        try:
            if fileformat == "sparky":
                sparky_str = self._to_sparky()
                filehandle.write(sparky_str)
            elif fileformat == "autoassign":
                autoassign_str = self._to_sparky()
                filehandle.write(autoassign_str)
            elif fileformat == "json":
                json_str = self._to_json()
                filehandle.write(json_str)
            else:
                raise TypeError("Unknown file format.")
        except IOError:
            raise IOError('"filehandle" parameter must be writable.')
        filehandle.close()