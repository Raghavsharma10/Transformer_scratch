def rowsAfterValue(self, value, count):
        """
        Retrieve some rows at or after a given sort-column value.

        @param value: Starting value in the index for the current sort column
        at which to start returning results.  Rows with a column value for the
        current sort column which is greater than or equal to this value will
        be returned.

        @type value: Some type compatible with the current sort column, or
        None, to specify the beginning of the data.

        @param count: The maximum number of rows to return.
        @type count: C{int}

        @return: A list of row data, ordered by the current sort column,
        beginning at C{value} and containing at most C{count} elements.
        """
        if value is None:
            query = self.inequalityQuery(None, count, True)
        else:
            pyvalue = self._toComparableValue(value)
            currentSortAttribute = self.currentSortColumn.sortAttribute()
            query = self.inequalityQuery(currentSortAttribute >= pyvalue, count, True)
        return self.constructRows(query)