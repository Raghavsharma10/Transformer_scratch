def enter(self):
        """Send a StartRequest."""
        self._communication.send(StartRequest,
                                 self._communication.left_end_needle,
                                 self._communication.right_end_needle)