def _remove(self, rtracker):
        """
        Remove a resource from the pool.

        :param rtracker: A resource.
        :type rtracker: :class:`_ResourceTracker`
        """
        with self._lock:
            i = self._reference_queue.index(rtracker)
            self._reference_queue[i] = None
            self._size -= 1