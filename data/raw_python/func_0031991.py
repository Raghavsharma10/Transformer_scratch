def process_forever(self, timeout=0.2):
        """Run an infinite loop, processing data from connections.

        This method repeatedly calls process_once.

        :param timeout: Parameter to pass to
                        :meth:`irc.client.Reactor.process_once`
        :type timeout: :class:`float`
        """
        # This loop should specifically *not* be mutex-locked.
        # Otherwise no other thread would ever be able to change
        # the shared state of a Reactor object running this function.
        log.debug("process_forever(timeout=%s)", timeout)
        self._looping.set()
        while self._looping.is_set():
            self.process_once(timeout)