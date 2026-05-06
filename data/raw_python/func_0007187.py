def write_cell(self, x, y, value):
        """
        Writing value in the cell of x+1 and y+1 position
        :param x: line index
        :param y: coll index
        :param value: value to be written
        :return:
        """
        x += 1
        y += 1
        self._sheet.update_cell(x, y, value)