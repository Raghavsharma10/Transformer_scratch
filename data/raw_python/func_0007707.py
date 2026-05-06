def add_row(self, data: list):
        """
        Add a row of buttons each with their own callbacks to the
        current widget.  Each element in `data` will consist of a
        label and a command.
        :param data: a list of tuples of the form ('label', <callback>)
        :return: None
        """

        # validation
        if self.headers and data:
            if len(self.headers) != len(data):
                raise ValueError

        offset = 0 if not self.headers else 1
        row = list()

        for i, e in enumerate(data):
            if not isinstance(e, tuple):
                raise ValueError('all elements must be a tuple '
                                 'consisting of ("label", <command>)')

            label, command = e
            button = tk.Button(self, text=str(label), relief=tk.RAISED,
                               command=command,
                               padx=self.padding,
                               pady=self.padding)

            button.grid(row=len(self._rows) + offset, column=i, sticky='ew')
            row.append(button)

        self._rows.append(row)