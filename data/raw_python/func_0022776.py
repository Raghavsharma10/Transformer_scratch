def associate(self, queue):
        """Merge this queue with another.

        Both queues will use a shared command list and either one can be used
        to fill or flush the shared queue.
        """
        assert isinstance(queue, GlirQueue)
        if queue._shared is self._shared:
            return

        # merge commands
        self._shared._commands.extend(queue.clear())
        self._shared._verbose |= queue._shared._verbose
        self._shared._associations[queue] = None
        # update queue and all related queues to use the same _shared object
        for ch in queue._shared._associations:
            ch._shared = self._shared
            self._shared._associations[ch] = None
        queue._shared = self._shared