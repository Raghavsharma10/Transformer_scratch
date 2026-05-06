def clear(self):
        """
        Clear out the multi-frame
        :return:
        """
        for slot in self._slots:
            slot.grid_forget()
            slot.destroy()

        self._slots = []