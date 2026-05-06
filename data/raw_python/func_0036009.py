def Queue(self, name, initial=None, maxsize=None):
        """The queue datatype.

        :param name: The name of the queue.
        :keyword initial: Initial items in the queue.

        See :class:`redish.types.Queue`.

        """
        return types.Queue(name, self.api, initial=initial, maxsize=maxsize)