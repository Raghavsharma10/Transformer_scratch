def write(self, *args, **kwargs):
        """
        :param args: tuple(value, style), tuple(value, style)
        :param kwargs: header=tuple(value, style), header=tuple(value, style)
        :param args: value, value
        :param kwargs: header=value, header=value
        """

        if args:
            kwargs = dict(zip(self.header, args))
        for header in kwargs:
            cell = kwargs[header]
            if not isinstance(cell, tuple):
                cell = (cell,)
            self.write_cell(self._row, self.header.index(header), *cell)
        self._row += 1