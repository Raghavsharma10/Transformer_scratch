def _run(self):
        """The inside of ``run``'s infinite loop.

        Separated out so it can be properly unit tested.
        """
        tup = self.read_tuple()
        self._current_tups = [tup]
        if self.is_heartbeat(tup):
            self.send_message({"command": "sync"})
        elif self.is_tick(tup):
            self.process_tick(tup)
            if self.auto_ack:
                self.ack(tup)
        else:
            self.process(tup)
            if self.auto_ack:
                self.ack(tup)
        # Reset _current_tups so that we don't accidentally fail the wrong
        # Tuples if a successive call to read_tuple fails.
        # This is not done in `finally` clause because we want the current
        # Tuples to fail when there is an exception.
        self._current_tups = []