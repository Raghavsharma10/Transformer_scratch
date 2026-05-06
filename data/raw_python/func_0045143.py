def process(self, sched, coro):
        """Add the given coroutine in the scheduler."""
        super(AddCoro, self).process(sched, coro)
        self.result = sched.add(self.coro, self.args, self.kwargs, self.prio & priority.OP)
        if self.prio & priority.CORO:
            return self, coro
        else:
            sched.active.append( (None, coro))