def _run(self):
        """The inside of ``run``'s infinite loop.

        Separate from BatchingBolt's implementation because
        we need to be able to acquire the batch lock after
        reading the tuple.

        We can't acquire the lock before reading the tuple because if
        that hangs (i.e. the topology is shutting down) the lock being
        acquired will freeze the rest of the bolt, which is precisely
        what this batcher seeks to avoid.
        """
        tup = self.read_tuple()
        with self._batch_lock:
            self._current_tups = [tup]
            if self.is_heartbeat(tup):
                self.send_message({"command": "sync"})
            elif self.is_tick(tup):
                self.process_tick(tup)
            else:
                self.process(tup)
            # reset so that we don't accidentally fail the wrong Tuples
            # if a successive call to read_tuple fails
            self._current_tups = []