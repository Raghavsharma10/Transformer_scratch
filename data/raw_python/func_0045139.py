def process(self, sched, coro):
        """Add the calling coro in a waiting for signal queue."""
        super(WaitForSignal, self).process(sched, coro)
        waitlist = sched.sigwait[self.name]
        waitlist.append((self, coro))
        if self.name in sched.signals:
            sig = sched.signals[self.name]
            if sig.recipients <= len(waitlist):
                sig.process(sched, sig.coro)
                del sig.coro
                del sched.signals[self.name]