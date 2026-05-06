def _get(self, timeout):
        """
        Get a resource from the pool. If timeout is ``None`` waits
        indefinitely.

        :param timeout: Time in seconds to wait for a resource.
        :type timeout: int
        :return: A resource.
        :rtype: :class:`_ResourceTracker`

        :raises PoolEmptyError: When timeout has elapsed and unable to
            retrieve resource.
        """
        with self._lock:
            if timeout is None:
                while self.empty():
                    self._not_empty.wait()
            else:
                time_end = time.time() + timeout
                while self.empty():
                    time_left = time_end - time.time()
                    if time_left < 0:
                        raise PoolEmptyError
                    self._not_empty.wait(time_left)

            rtracker = self._reference_queue[self._resource_start]
            self._resource_start = (self._resource_start + 1) % self.maxsize
            self._available -= 1

        return rtracker