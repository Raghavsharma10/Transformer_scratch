def finish(self, msg=None):
        """Add a final message; flush the message list if no parent profiler.
        """
        if self._finished or self.disable:
            return        
        self._finished = True
        if msg is not None:
            self(msg)
        self._new_msg("< Exiting %s, total time: %0.4f ms", 
                      self._name, (ptime.time() - self._firstTime) * 1000)
        type(self)._depth -= 1
        if self._depth < 1:
            self.flush()