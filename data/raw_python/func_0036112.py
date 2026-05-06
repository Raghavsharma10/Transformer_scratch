def update(self, other):
        """Update this set with the union of itself and others."""
        if isinstance(other, self.__class__):
            return self.client.sunionstore(self.name, [self.name, other.name])
        else:
            return map(self.add, other)