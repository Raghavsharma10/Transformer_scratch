def remove(self, member):
        """Remove member."""
        if not self.client.zrem(self.name, member):
            raise KeyError(member)