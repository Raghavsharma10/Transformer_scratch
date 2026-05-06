def channel(self):
        """If no channel exists, a new one is requested."""
        if not self._channel:
            self._channel_ref = weakref.ref(self.connection.get_channel())
        return self._channel