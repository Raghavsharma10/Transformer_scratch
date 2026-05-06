def cleanup(self, sched, coro):
        """Remove this coro from the waiting for signal queue."""
        try:
            sched.sigwait[self.name].remove((self, coro))
        except ValueError:
            pass
        return True