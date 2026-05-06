def _redraw(self):
        """
        Forgets the current layout and redraws with the most recent information

        :return: None
        """
        for row in self._rows:
            for widget in row:
                widget.grid_forget()

        offset = 0 if not self.headers else 1
        for i, row in enumerate(self._rows):
            for j, widget in enumerate(row):
                widget.grid(row=i+offset, column=j)