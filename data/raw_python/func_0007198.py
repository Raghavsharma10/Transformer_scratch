def set_item(self, index, new_item):
        """ Changes item at index in collection. Emit dataChanged signal.

        :param index: Number of row or index of cell
        :param new_item: Dict-like object
        """
        row = index.row() if hasattr(index, "row") else index
        self.collection[row] = new_item
        self.dataChanged.emit(self.index(
            row, 0), self.index(row, self.rowCount() - 1))