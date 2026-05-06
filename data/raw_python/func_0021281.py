def _recycle(self):
        """ Reclaim buffer space before the origin.

        Note: modifies buffer size
        """
        origin = self._origin
        if origin == 0:
            return False
        available = self._extent - origin
        self._data[:available] = self._data[origin:self._extent]
        self._extent = available
        self._origin = 0
        #log_debug("Recycled %d bytes" % origin)
        return True