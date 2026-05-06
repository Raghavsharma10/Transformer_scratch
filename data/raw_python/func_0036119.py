def increment(self, member, amount=1):
        """Increment the score of ``member`` by ``amount``."""
        return self.client.zincrby(self.name, member, amount)