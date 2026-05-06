def shrink(self):
        "Get rid of one worker from the pool. Raises IndexError if empty."
        if self._size <= 0:
            raise IndexError("pool is already empty")
        self._size -= 1
        self.put(SuicideJob())