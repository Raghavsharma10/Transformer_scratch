def discard_incoming_messages(self):
        """
        Discard all incoming messages for the time of the context manager.
        """
        # Flush any received message so far.
        self.inbox.clear()

        # This allows nesting of discard_incoming_messages() calls.
        previous = self._discard_incoming_messages
        self._discard_incoming_messages = True

        try:
            yield
        finally:
            self._discard_incoming_messages = previous