def listen(self, you):
        """
        Request a callback for value modification.

        Parameters
        ----------
        you : object
            An instance having ``__call__`` attribute.
        """
        self._listeners.append(you)
        self.raw.talk_to(you)