def difference_update(self, other):
        """Remove all elements of another set from this set."""
        return self.client.sdiffstore(self.name, [self.name, other.name])