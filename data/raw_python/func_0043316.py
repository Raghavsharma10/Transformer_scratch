async def write(self, item):
        """
        Write an item in the queue.

        :param item: The item.
        """
        await self._queue.put(item)
        self._can_read.set()

        if self._queue.full():
            self._can_write.clear()