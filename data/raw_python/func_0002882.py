def data(self, filtered=False, no_empty=False):
        r"""
         Return (filtered) file data.

         The returned object is a list, each item is a sub-list corresponding
         to a row of data; each item in the sub-lists contains data
         corresponding to a particular column

        :param filtered: Filtering type
        :type  filtered: :ref:`CsvFiltered`

        :param no_empty: Flag that indicates whether rows with empty columns
                         should be filtered out (True) or not (False)
        :type  no_empty: bool

        :rtype: list

        .. [[[cog cog.out(exobj.get_sphinx_autodoc()) ]]]
        .. Auto-generated exceptions documentation for
        .. pcsv.csv_file.CsvFile.data

        :raises:
         * RuntimeError (Argument \`filtered\` is not valid)

         * RuntimeError (Argument \`no_empty\` is not valid)

        .. [[[end]]]
        """
        return self._apply_filter(filtered, no_empty)