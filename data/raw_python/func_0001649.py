def _get_queue(self):
        """Gets the actual location of the queue, or None.
        """
        if self._queue is None:
            self._links = []
            queue, depth = self._resolve_queue(self.queue, links=self._links)
            if queue is None and depth > 0:
                raise QueueLinkBroken
            self._queue = queue
        return self._queue