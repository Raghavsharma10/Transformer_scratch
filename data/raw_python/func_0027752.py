def _massageData(self, row):
        """
        Convert a row into an Item instance by loading cached items or
        creating new ones based on query results.

        @param row: an n-tuple, where n is the number of columns specified by
        my item type.

        @return: an instance of the type specified by this query.
        """
        result = self.store._loadedItem(self.tableClass, row[0], row[1:])
        assert result.store is not None, "result %r has funky store" % (result,)
        return result