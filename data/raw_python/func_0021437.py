def flags(self, index):
        """Return the active flags for the given index. Add editable
        flag to items other than the first column.
        """
        activeFlags = (Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                       Qt.ItemIsUserCheckable)

        item = self.item(index)
        column = index.column()

        if column > 0 and not item.childCount():
            activeFlags = activeFlags | Qt.ItemIsEditable

        return activeFlags