def add_row(self, data: list=None):
        """
        Add a row of data to the current widget, add a <Tab> \
        binding to the last element of the last row, and set \
        the focus at the beginning of the next row.

        :param data: a row of data
        :return: None
        """
        # validation
        if self.headers and data:
            if len(self.headers) != len(data):
                raise ValueError

        offset = 0 if not self.headers else 1
        row = list()

        if data:
            for i, element in enumerate(data):
                contents = '' if element is None else str(element)
                entry = ttk.Entry(self)
                entry.insert(0, contents)
                entry.grid(row=len(self._rows) + offset,
                           column=i,
                           sticky='E,W')
                row.append(entry)
        else:
            for i in range(self.num_of_columns):
                entry = ttk.Entry(self)
                entry.grid(row=len(self._rows) + offset,
                           column=i,
                           sticky='E,W')
                row.append(entry)

        self._rows.append(row)

        # clear all bindings
        for row in self._rows:
            for widget in row:
                widget.unbind('<Tab>')

        def add(e):
            self.add_row()

        last_entry = self._rows[-1][-1]
        last_entry.bind('<Tab>', add)

        e = self._rows[-1][0]
        e.focus_set()

        self._redraw()