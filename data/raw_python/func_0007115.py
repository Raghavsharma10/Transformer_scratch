def write_cell(self, x, y, value, style=None):
        """
            writing style and value in the cell of x and y position
        """
        if isinstance(style, str):
            style = self.xlwt.easyxf(style)
        if style:
            self._sheet.write(x, y, label=value, style=style)
        else:
            self._sheet.write(x, y, label=value)