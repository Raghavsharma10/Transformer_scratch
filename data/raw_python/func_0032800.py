def _sortAttributeValue(self, offset):
        """
        return the value of the sort attribute for the item at
        'offset' in the results of the last query, otherwise None.
        """
        if self._currentResults:
            pageStart = (self._currentResults[offset][
                self.currentSortColumn.attributeID],
                         self._currentResults[offset][
                    '__item__'].storeID)
        else:
            pageStart = None
        return pageStart