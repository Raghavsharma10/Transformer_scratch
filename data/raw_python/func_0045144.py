def process(self, sched, coro):
        """Add the calling coroutine as a waiter in the coro we want to join.
        Also, doesn't keep the called active (we'll be activated back when the
        joined coro dies)."""
        super(Join, self).process(sched, coro)
        self.coro.add_waiter(coro)