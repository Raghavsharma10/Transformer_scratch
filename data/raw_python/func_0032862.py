def rowsBeforeItem(self, item, count):
        """
        The inverse of rowsAfterItem.

        @param item: then L{Item} to request rows before.
        @type item: this L{InequalityModel}'s L{itemType} attribute.

        @param count: The maximum number of rows to return.
        @type count: L{int}

        @return: A list of row data, ordered by the current sort column,
        beginning immediately after C{item}.
        """
        currentSortAttribute = self.currentSortColumn.sortAttribute()
        value = currentSortAttribute.__get__(item, type(item))
        firstQuery = self.inequalityQuery(
            AND(currentSortAttribute == value,
                self.itemType.storeID < item.storeID),
            count, False)
        results = self.constructRows(firstQuery)
        count -= len(results)
        if count:
            secondQuery = self.inequalityQuery(currentSortAttribute < value,
                                               count, False)
            results.extend(self.constructRows(secondQuery))
        return results[::-1]