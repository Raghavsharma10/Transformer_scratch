def Set(self, name, initial=None):
        """The set datatype.

        :param name: The name of the set.
        :keyword initial: Initial members of the set.

        See :class:`redish.types.Set`.

        """
        return types.Set(name, self.api, initial)