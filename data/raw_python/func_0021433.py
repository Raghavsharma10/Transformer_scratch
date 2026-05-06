def parent(self, index):
        """Return the index of the parent for a given index of the
        child. Unfortunately, the name of the method has to be parent,
        even though a more verbose name like parentIndex, would avoid
        confusion about what parent actually is - an index or an item.
        """
        childItem = self.item(index)
        parentItem = childItem.parent

        if parentItem == self.rootItem:
            parentIndex = QModelIndex()
        else:
            parentIndex = self.createIndex(parentItem.row(), 0, parentItem)

        return parentIndex