def setData(self, index, value, role):
        """Set the role data for the item at index to value."""
        if not index.isValid():
            return False

        item = self.item(index)
        column = index.column()

        if role == Qt.EditRole:
            items = list()
            items.append(item)

            if self.sync:
                parentIndex = self.parent(index)
                # Iterate over the siblings of the parent index.
                for sibling in self.siblings(parentIndex):
                    siblingNode = self.item(sibling)
                    for child in siblingNode.children:
                        if child.getItemData(0) == item.getItemData(0):
                            items.append(child)

            for item in items:
                columnData = str(item.getItemData(column))
                if columnData and columnData != value:
                    try:
                        item.setItemData(column, float(value))
                    except ValueError:
                        return False
                else:
                    return False

        elif role == Qt.CheckStateRole:
            item.setCheckState(value)
            if value == Qt.Unchecked or value == Qt.Checked:
                state = value
                self.itemCheckStateChanged.emit(index, state)

        self.dataChanged.emit(index, index)

        return True