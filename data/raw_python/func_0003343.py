def start(self, asyncStart = False):
        """
        Start `container.main` as the main routine.
        
        :param asyncStart: if True, start the routine in background. By default, the routine
                           starts in foreground, which means it is executed to the first
                           `yield` statement before returning. If the started routine raises
                           an exception, the exception is re-raised to the caller of `start`
        """
        r = Routine(self.main(), self.scheduler, asyncStart, self, True, self.daemon)
        self.mainroutine = r
        try:
            next(r)
        except StopIteration:
            pass
        return r