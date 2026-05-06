def index(self, row, column, parent=QModelIndex()):
        """Return the index of the item in the model specified by the
        given row, column, and parent index.
        """
        if parent is not None and not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = self.item(parent)

        childItem = parentItem.child(row)

        if childItem:
            index = self.createIndex(row, column, childItem)
        else:
            index = QModelIndex()

        return index