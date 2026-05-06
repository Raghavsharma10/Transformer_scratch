def union(self, other):
        """Return the union of sets as a new set.

        (i.e. all elements that are in either set.)

        Operates on either redish.types.Set or __builtins__.set.

        """
        if isinstance(other, self.__class__):
            return self.client.sunion([self.name, other.name])
        else:
            return self._as_set().union(other)