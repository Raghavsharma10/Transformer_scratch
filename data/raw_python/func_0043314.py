async def read(self):
        """
        Read from the box in a blocking manner.

        :returns: An item from the box.
        """
        result = await self._queue.get()

        self._can_write.set()

        if self._queue.empty():
            self._can_read.clear()

        return result