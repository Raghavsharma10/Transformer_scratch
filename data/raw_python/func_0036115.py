def difference(self, *others):
        """Return the difference of two or more sets as a new :class:`set`.

        (i.e. all elements that are in this set but not the others.)

        Operates on either redish.types.Set or __builtins__.set.

        """
        if all([isinstance(a, self.__class__) for a in others]):
            return self.client.sdiff([self.name] + [other.name for other in others])
        else:
            othersets = filter(lambda x: isinstance(x, set), others)
            otherTypes = filter(lambda x: isinstance(x, self.__class__), others)
            return self.client.sdiff([self.name] + [other.name for other in otherTypes]).difference(*othersets)