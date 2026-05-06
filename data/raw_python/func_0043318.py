def clone(self):
        """
        Clone the box.

        :returns: A new box with the same item queue.

        The cloned box is not closed, no matter the initial state of the
        original instance.
        """
        result = AsyncBox(maxsize=self._maxsize, loop=self.loop)
        result._queue = self._queue
        result._can_read = self._can_read
        result._can_write = self._can_write

        return result