def requestRowRange(self, rangeBegin, rangeEnd):
        """
        Retrieve display data for the given range of rows.

        @type rangeBegin: C{int}
        @param rangeBegin: The index of the first row to retrieve.

        @type rangeEnd: C{int}
        @param rangeEnd: The index of the last row to retrieve.

        @return: A C{list} of C{dict}s giving row data.
        """
        return self.constructRows(self.performQuery(rangeBegin, rangeEnd))