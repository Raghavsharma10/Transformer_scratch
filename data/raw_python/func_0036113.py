def intersection(self, other):
        """Return the intersection of two sets as a new set.

        (i.e. all elements that are in both sets.)

        Operates on either redish.types.Set or __builtins__.set.

        """
        if isinstance(other, self.__class__):
            return self.client.sinter([self.name, other.name])
        else:
            return self._as_set().intersection(other)