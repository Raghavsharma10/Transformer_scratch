def read_cell(self, x, y):
        """
        Reads the cell at position x+1 and y+1; return value
        :param x: line index
        :param y: coll index
        :return: {header: value}
        """
        if isinstance(self.header[y], tuple):
            header = self.header[y][0]
        else:
            header = self.header[y]
        x += 1
        y += 1
        if self.strip:
            self._sheet.cell(x, y).value = self._sheet.cell(x, y).value.strip()
        else:
            return {header: self._sheet.cell(x, y).value}