def List(self, name, initial=None):
        """The list datatype.

        :param name: The name of the list.
        :keyword initial: Initial contents of the list.

        See :class:`redish.types.List`.

        """
        return types.List(name, self.api, initial=initial)