def _put(self, rtracker):
        """
        Put a resource back in the queue.

        :param rtracker: A resource.
        :type rtracker: :class:`_ResourceTracker`

        :raises PoolFullError: If pool is full.
        :raises UnknownResourceError: If resource can't be found.
        """
        with self._lock:
            if self._available < self.capacity:
                for i in self._unavailable_range():
                    if self._reference_queue[i] is rtracker:
                        # i retains its value and will be used to swap with
                        # first "empty" space in queue.
                        break
                else:
                    raise UnknownResourceError

                j = self._resource_end
                rq = self._reference_queue
                rq[i], rq[j] = rq[j], rq[i]

                self._resource_end = (self._resource_end + 1) % self.maxsize
                self._available += 1

                self._not_empty.notify()
            else:
                raise PoolFullError