def header(self, filtered=False):
        r"""
        Return data header.

        When the raw (input) data is used the data header is a list of the
        comma-separated values file header if the file is loaded with header
        (each list item is a column header) or a list of column numbers if the
        file is loaded without header (column zero is the leftmost column).
        When filtered data is used the data header is the active column filter,
        if any, otherwise it is the same as the raw (input) data header

        :param filtered: Flag that indicates whether the raw (input) data
                         should be used (False) or whether filtered data
                         should be used (True)
        :type  filtered: boolean

        :rtype: list of strings or integers

        .. [[[cog cog.out(exobj.get_sphinx_autodoc()) ]]]
        .. Auto-generated exceptions documentation for
        .. pcsv.csv_file.CsvFile.header

        :raises: RuntimeError (Argument \`filtered\` is not valid)

        .. [[[end]]]
        """
        return (
            self._header
            if (not filtered) or (filtered and self._cfilter is None)
            else self._cfilter
        )