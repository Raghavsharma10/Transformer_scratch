def add(self, string: (str, list)):
        """
        Add a new slot to the multi-frame containing the string.
        :param string: a string to insert
        :return: None
        """
        slot = _SlotFrame(self,
                          remove_callback=self._redraw,
                          entries=self._slot_columns)
        slot.add(string)

        self._slots.append(slot)

        self._redraw()