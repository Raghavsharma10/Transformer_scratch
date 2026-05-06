def subroutine(self, iterator, asyncStart = True, name = None, daemon = False):
        """
        Start extra routines in this container.
        
        :param iterator: A coroutine object i.e the return value of an async method `my_routine()`
        
        :param asyncStart: if False, start the routine in foreground. By default, the routine
                           starts in background, which means it is not executed until the current caller
                           reaches the next `yield` statement or quit.
        
        :param name: if not None, `container.<name>` is set to the routine object. This is useful when
                     you want to terminate the routine from outside.
                     
        :param daemon: if True, this routine is set to be a daemon routine.
                       A daemon routine does not stop the scheduler from quitting; if all non-daemon routines
                       are quit, the scheduler stops. 
        """
        r = Routine(iterator, self.scheduler, asyncStart, self, True, daemon)
        if name is not None:
            setattr(self, name, r)
        # Call subroutine may change the currentroutine, we should restore it
        currentroutine = getattr(self, 'currentroutine', None)
        try:
            next(r)
        except StopIteration:
            pass
        self.currentroutine = currentroutine
        return r