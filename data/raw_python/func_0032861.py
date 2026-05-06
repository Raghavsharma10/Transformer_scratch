def rowsBeforeValue(self, value, count):
        """
        Retrieve display data for rows with sort-column values less than the
        given value.

        @type value: Some type compatible with the current sort column.
        @param value: Starting value in the index for the current sort column
        at which to start returning results.  Rows with a column value for the
        current sort column which is less than this value will be returned.

        @type count: C{int}
        @param count: The number of rows to return.

        @return: A list of row data, ordered by the current sort column, ending
        at C{value} and containing at most C{count} elements.
        """
        if value is None:
            query = self.inequalityQuery(None, count, False)
        else:
            pyvalue = self._toComparableValue(value)
            currentSortAttribute = self.currentSortColumn.sortAttribute()
            query = self.inequalityQuery(
                currentSortAttribute < pyvalue, count, False)
        return self.constructRows(query)[::-1]