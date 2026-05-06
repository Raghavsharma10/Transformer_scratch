def process(self, sched, coro):
        """Add the timeout in the scheduler, check for defaults."""
        super(TimedOperation, self).process(sched, coro)

        if sched.default_timeout and not self.timeout:
            self.set_timeout(sched.default_timeout)
        if self.timeout and self.timeout != -1:
            self.coro = coro

            if self.weak_timeout:
                self.last_checkpoint = getnow()
                self.delta = self.timeout - self.last_checkpoint
            else:
                self.last_checkpoint = self.delta = None

            heapq.heappush(sched.timeouts, self)