def syscall_noreturn(self, func):
        '''
        Call a syscall method. A syscall method is executed outside of any routines, directly
        in the scheduler loop, which gives it chances to directly operate the event loop.
        See :py:method::`vlcp.event.core.Scheduler.syscall`.
        '''
        matcher = self.scheduler.syscall(func)
        while not matcher:
            yield
            matcher = self.scheduler.syscall(func)
        ev, _ = yield (matcher,)
        return ev