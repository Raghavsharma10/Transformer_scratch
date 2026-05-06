def process(self, sched, coro):
        """This is called when the operation is to be processed by the
        scheduler. Code here works modifies the scheduler and it's usualy
        very crafty. Subclasses usualy overwrite this method and call it from
        the superclass."""

        if self.prio == priority.DEFAULT:
            self.prio = sched.default_priority