def get_item(self, index):
        """ Acces shortcut

        :param index: Number of row or index of cell
        :return: Dict-like item
        """
        row = index.row() if hasattr(index, "row") else index
        try:
            return self.collection[row]
        except IndexError:  # invalid index for exemple
            return None