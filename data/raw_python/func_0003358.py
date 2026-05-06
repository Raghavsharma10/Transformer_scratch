def begin_delegate_other(self, subprocess, container, retnames = ('',)):
        '''
        DEPRECATED Start the delegate routine, but do not wait for result, instead returns a (matcher routine) tuple.
        Useful for advanced delegates (e.g. delegate multiple subprocesses in the same time).
        This is NOT a coroutine method.
        
        :param subprocess: a coroutine
        
        :param container: container in which to start the routine
        
        :param retnames: get return values from keys. '' for the return value (for compatibility with earlier versions)
        
        :returns: (matcher, routine) where matcher is a event matcher to get the delegate result, routine is the created routine
        '''
        async def delegateroutine():
            try:
                r = await subprocess
            except:
                _, val, _ = sys.exc_info()
                e = RoutineControlEvent(RoutineControlEvent.DELEGATE_FINISHED, container.currentroutine, exception = val)
                container.scheduler.emergesend(e)
                raise
            else:
                e = RoutineControlEvent(RoutineControlEvent.DELEGATE_FINISHED, container.currentroutine,
                                        result = tuple(r if n == '' else getattr(container, n, None)
                                                       for n in retnames))
                await container.waitForSend(e)
        r = container.subroutine(generatorwrapper(delegateroutine(), 'subprocess', 'delegate'), True)
        return (RoutineControlEvent.createMatcher(RoutineControlEvent.DELEGATE_FINISHED, r), r)