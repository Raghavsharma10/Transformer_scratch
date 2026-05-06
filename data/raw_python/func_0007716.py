def get(self):
        """
        Returns the value for the slot.
        :return: the entry value
        """
        values = [e.get() for e in self._entries]
        if len(self._entries) == 1:
            return values[0]
        else:
            return values