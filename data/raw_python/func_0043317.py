def write_nowait(self, item):
        """
        Write in the box in a non-blocking manner.

        If the box is full, an exception is thrown. You should always check
        for fullness with `full` or `wait_not_full` before calling this method.

        :param item: An item.
        """
        self._queue.put_nowait(item)
        self._can_read.set()

        if self._queue.full():
            self._can_write.clear()