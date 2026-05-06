def _redraw(self):
        """
        Clears the current layout and re-draws all elements in self._slots
        :return:
        """
        if self._blank_label:
            self._blank_label.grid_forget()
            self._blank_label.destroy()
            self._blank_label = None

        for slot in self._slots:
            slot.grid_forget()

        self._slots = [slot for slot in self._slots if not slot.deleted]

        max_per_col = 8
        for i, slot in enumerate(self._slots):
            slot.grid(row=i % max_per_col,
                      column=int(i / max_per_col), sticky='ew')