def put(self, item, **kwargs):
        """Put an item into the queue."""
        if self.full():
            raise Full()
        self._append(item)