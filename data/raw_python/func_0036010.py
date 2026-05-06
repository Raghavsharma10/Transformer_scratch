def LifoQueue(self, name, initial=None, maxsize=None):
        """The LIFO queue datatype.

        :param name: The name of the queue.
        :keyword initial: Initial items in the queue.

        See :class:`redish.types.LifoQueue`.

        """
        return types.LifoQueue(name, self.api,
                               initial=initial, maxsize=maxsize)