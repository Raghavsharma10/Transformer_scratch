def get(self, timeout=None):
        """Return the next available item from the tube.

        Blocks if tube is empty, until a producer for the tube puts an item on it."""
        if timeout:
            # Todo: Consider locking the poll/recv block.
            # Otherwise, this method is not thread safe.
            if self._conn1.poll(timeout):
                return (True, self._conn1.recv())
            else:
                return (False, None)
        return self._conn1.recv()