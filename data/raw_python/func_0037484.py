def write(self, filehandle, file_format):
        """Write :class:`~nmrstarlib.nmrstarlib.StarFile` data into file.

        :param filehandle: file-like object.
        :type filehandle: :py:class:`io.TextIOWrapper`
        :param str file_format: Format to use to write data: `nmrstar`, `cif`, or `json`.
        :return: None
        :rtype: :py:obj:`None`
        """
        try:
            if file_format == "json":
                json_str = self._to_json()
                filehandle.write(json_str)
            elif file_format == "nmrstar" and isinstance(self, NMRStarFile):
                nmrstar_str = self._to_star()
                filehandle.write(nmrstar_str)
            elif file_format == "cif" and isinstance(self, CIFFile):
                cif_str = self._to_star()
                filehandle.write(cif_str)
            else:
                raise TypeError("Unknown file format.")
        except IOError:
            raise IOError('"filehandle" parameter must be writable.')
        filehandle.close()