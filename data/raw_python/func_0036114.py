def intersection_update(self, other):
        """Update the set with the intersection of itself and another."""
        return self.client.sinterstore(self.name, [self.name, other.name])