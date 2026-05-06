def replace(self, rdata, filtered=False):
        r"""
        Replace data.

        :param rdata: Replacement data
        :type  rdata: list of lists

        :param filtered: Filtering type
        :type  filtered: :ref:`CsvFiltered`

        .. [[[cog cog.out(exobj.get_sphinx_autodoc(width=63)) ]]]
        .. Auto-generated exceptions documentation for
        .. pcsv.csv_file.CsvFile.replace

        :raises:
         * RuntimeError (Argument \`filtered\` is not valid)

         * RuntimeError (Argument \`rdata\` is not valid)

         * ValueError (Number of columns mismatch between input and
           replacement data)

         * ValueError (Number of rows mismatch between input and
           replacement data)

        .. [[[end]]]
        """
        # pylint: disable=R0914
        rdata_ex = pexdoc.exh.addai("rdata")
        rows_ex = pexdoc.exh.addex(
            ValueError, "Number of rows mismatch between input and replacement data"
        )
        cols_ex = pexdoc.exh.addex(
            ValueError, "Number of columns mismatch between input and replacement data"
        )
        rdata_ex(any([len(item) != len(rdata[0]) for item in rdata]))
        # Use all columns if no specification has been given
        cfilter = (
            self._cfilter if filtered in [True, "B", "b", "C", "c"] else self._header
        )
        # Verify column names, has to be done before getting data
        col_num = len(self._data[0]) - 1
        odata = self._apply_filter(filtered)
        cfilter = (
            self._cfilter if filtered in [True, "B", "b", "C", "c"] else self._header
        )
        col_index = [
            self._header_upper.index(col_id.upper())
            if isinstance(col_id, str)
            else col_id
            for col_id in cfilter
        ]
        rows_ex(len(odata) != len(rdata))
        cols_ex(len(odata[0]) != len(rdata[0]))
        df_tuples = self._format_rfilter(self._rfilter)
        rnum = 0
        for row in self._data:
            if (not filtered) or (
                filtered
                and all([row[col_num] in col_value for col_num, col_value in df_tuples])
            ):
                for col_num, new_data in zip(col_index, rdata[rnum]):
                    row[col_num] = new_data
                rnum = rnum + 1