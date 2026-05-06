def data(self, index, role):
        """Return role specific data for the item referred by
        index.column()."""
        if not index.isValid():
            return

        item = self.item(index)
        column = index.column()
        value = item.getItemData(column)

        if role == Qt.DisplayRole:
            try:
                if column == 1:
                    # Display small values using scientific notation.
                    if abs(float(value)) < 1e-3 and float(value) != 0.0:
                        return '{0:8.1e}'.format(value)
                    else:
                        return '{0:8.3f}'.format(value)
                else:
                    return '{0:8.2f}'.format(value)
            except ValueError:
                return value
        elif role == Qt.EditRole:
            try:
                value = float(value)
                if abs(value) < 1e-3 and value != 0.0:
                    return str('{0:8.1e}'.format(value))
                else:
                    return str('{0:8.3f}'.format(value))
            except ValueError:
                return str(value)
        elif role == Qt.CheckStateRole:
            if item.parent == self.rootItem and column == 0:
                return item.getCheckState()
        elif role == Qt.TextAlignmentRole:
            if column > 0:
                return Qt.AlignRight