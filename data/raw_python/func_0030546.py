def state(self, value):
        """set player state, emit state changed signal

        outer object should not set state directly,
        use ``pause`` / ``resume`` / ``stop`` / ``play`` method instead.
        """
        self._state = value
        self.state_changed.emit(value)