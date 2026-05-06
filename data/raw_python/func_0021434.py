def rowCount(self, parentIndex):
        """Return the number of rows under the given parent. When the
        parentIndex is valid, rowCount() returns the number of children
        of the parent. For this it uses item() method to extract the
        parentItem from the parentIndex, and calls the childCount() of
        the item to get number of children.
        """
        if parentIndex.column() > 0:
            return 0

        if not parentIndex.isValid():
            parentItem = self.rootItem
        else:
            parentItem = self.item(parentIndex)

        return parentItem.childCount()