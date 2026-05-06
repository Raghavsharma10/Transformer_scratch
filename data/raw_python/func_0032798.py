def resort(self, attributeID, isAscending=None):
        """Sort by one of my specified columns, identified by attributeID
        """
        if isAscending is None:
            isAscending = self.defaultSortAscending

        newSortColumn = self.columns[attributeID]
        if newSortColumn.sortAttribute() is None:
            raise Unsortable('column %r has no sort attribute' % (attributeID,))
        if self.currentSortColumn == newSortColumn:
            # if this query is to be re-sorted on the same column, but in the
            # opposite direction to our last query, then use the first item in
            # the result set as the marker
            if self.isAscending == isAscending:
                offset = 0
            else:
                # otherwise use the last
                offset = -1
        else:
            offset = 0
            self.currentSortColumn = newSortColumn
        self.isAscending = isAscending
        self._updateResults(self._sortAttributeValue(offset), True)