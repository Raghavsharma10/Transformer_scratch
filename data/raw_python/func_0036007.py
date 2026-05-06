def SortedSet(self, name, initial=None):
        """The sorted set datatype.

        :param name: The name of the sorted set.
        :param initial: Initial members of the set as an iterable
           of ``(element, score)`` tuples.

        See :class:`redish.types.SortedSet`.

        """
        return types.SortedSet(name, self.api, initial)