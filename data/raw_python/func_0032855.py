def resort(self, columnName):
        """
        Re-sort the table.

        @param columnName: the name of the column to sort by.  This is a string
        because it is passed from the browser.
        """
        csc = self.currentSortColumn
        newSortColumn = self.columns[columnName]
        if newSortColumn is None:
            raise Unsortable('column %r has no sort attribute' % (columnName,))
        if csc is newSortColumn:
            self.isAscending = not self.isAscending
        else:
            self.currentSortColumn = newSortColumn
            self.isAscending = True
        return self.isAscending