def increment(self, member, amount=1):
        """Increment the score of ``member`` by ``amount``."""
        self._dict[member] += amount
        return self._dict[member]