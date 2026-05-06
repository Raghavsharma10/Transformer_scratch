def add_row(self, data: list):
        """
        Add a row of data to the current widget

        :param data: a row of data
        :return: None
        """
        # validation
        if self.headers:
            if len(self.headers) != len(data):
                raise ValueError

        if len(data) != self.num_of_columns:
            raise ValueError

        offset = 0 if not self.headers else 1
        row = list()
        for i, element in enumerate(data):
            label = ttk.Label(self, text=str(element), relief=tk.GROOVE,
                              padding=self.padding)
            label.grid(row=len(self._rows) + offset, column=i, sticky='E,W')
            row.append(label)

        self._rows.append(row)