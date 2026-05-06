def remove(self):
        """
        Deletes itself.
        :return: None
        """
        for e in self._entries:
            e.grid_forget()
            e.destroy()

        self._remove_btn.grid_forget()
        self._remove_btn.destroy()

        self.deleted = True

        if self._remove_callback:
            self._remove_callback()