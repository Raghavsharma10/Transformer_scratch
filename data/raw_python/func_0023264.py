def resize_bytes(self, size):
        """ Resize the buffer (in-place, deferred operation)

        Parameters
        ----------
        size : integer
            New buffer size in bytes

        Notes
        -----
        This clears any pending operations.
        """
        Buffer.resize_bytes(self, size)
        self._size = size // self.itemsize