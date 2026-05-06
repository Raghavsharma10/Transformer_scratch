def get_event(self, block=True, timeout=None):
        """ Get the next event from the queue.

        :arg boolean block: Set to True to block if no event is available.
        :arg seconds timeout: Timeout to wait if no event is available.

        :Returns: The next event as a :class:`pygerrit.events.GerritEvent`
            instance, or `None` if:
             - `block` is False and there is no event available in the queue, or
             - `block` is True and no event is available within the time
               specified by `timeout`.

        """
        try:
            return self._events.get(block, timeout)
        except Empty:
            return None