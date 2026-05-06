def Dict(self, name, initial=None, **extra):
        """The dictionary datatype (Hash).

        :param name: The name of the dictionary.
        :keyword initial: Initial contents.
        :keyword \*\*extra: Initial contents as keyword arguments.

        The ``initial``, and ``**extra`` keyword arguments
        will be merged (keyword arguments has priority).

        See :class:`redish.types.Dict`.

        """
        return types.Dict(name, self.api, initial=initial, **extra)