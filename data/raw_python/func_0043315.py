def read_nowait(self):
        """
        Read from the box in a non-blocking manner.

        If the box is empty, an exception is thrown. You should always check
        for emptiness with `empty` or `wait_not_empty` before calling this
        method.

        :returns: An item from the box.
        """
        result = self._queue.get_nowait()

        self._can_write.set()

        if self._queue.empty():
            self._can_read.clear()

        return result