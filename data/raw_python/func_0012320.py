def rehash(self, password):
        """Recreates the internal hash."""
        self.hash = self._new(password, self.desired_rounds)
        self.rounds = self.desired_rounds