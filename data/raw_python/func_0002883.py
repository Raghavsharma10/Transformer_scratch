def dsort(self, order):
        r"""
        Sort rows.

        :param order: Sort order
        :type  order: :ref:`CsvColFilter`

        .. [[[cog cog.out(exobj.get_sphinx_autodoc()) ]]]
        .. Auto-generated exceptions documentation for
        .. pcsv.csv_file.CsvFile.dsort

        :raises:
         * RuntimeError (Argument \`order\` is not valid)

         * RuntimeError (Invalid column specification)

         * ValueError (Column *[column_identifier]* not found)

        .. [[[end]]]
        """
        # Make order conforming to a list of dictionaries
        order = order if isinstance(order, list) else [order]
        norder = [{item: "A"} if not isinstance(item, dict) else item for item in order]
        # Verify that all columns exist in file
        self._in_header([list(item.keys())[0] for item in norder])
        # Get column indexes
        clist = []
        for nitem in norder:
            for key, value in nitem.items():
                clist.append(
                    (
                        key
                        if isinstance(key, int)
                        else self._header_upper.index(key.upper()),
                        value.upper() == "D",
                    )
                )
        # From the Python documentation:
        # "Starting with Python 2.3, the sort() method is guaranteed to be
        # stable. A sort is stable if it guarantees not to change the
        # relative order of elements that compare equal - this is helpful
        # for sorting in multiple passes (for example, sort by department,
        # then by salary grade)."
        # This means that the sorts have to be done from "minor" column to
        # "major" column
        for (cindex, rvalue) in reversed(clist):
            fpointer = operator.itemgetter(cindex)
            self._data.sort(key=fpointer, reverse=rvalue)