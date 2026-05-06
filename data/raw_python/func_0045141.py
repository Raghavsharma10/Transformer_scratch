def process(self, sched, coro):
        """If there aren't enough coroutines waiting for the signal as the
        recipicient param add the calling coro in another queue to be activated
        later, otherwise activate the waiting coroutines."""
        super(Signal, self).process(sched, coro)
        self.result = len(sched.sigwait[self.name])
        if self.result < self.recipients:
            sched.signals[self.name] = self
            self.coro = coro
            return

        for waitop, waitcoro in sched.sigwait[self.name]:
            waitop.result = self.value
        if self.prio & priority.OP:
            sched.active.extendleft(sched.sigwait[self.name])
        else:
            sched.active.extend(sched.sigwait[self.name])

        if self.prio & priority.CORO:
            sched.active.appendleft((None, coro))
        else:
            sched.active.append((None, coro))

        del sched.sigwait[self.name]