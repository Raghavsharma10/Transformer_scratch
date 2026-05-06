def begin_delegate(self, subprocess):
        '''
        Start the delegate routine, but do not wait for result, instead returns a (matcher, routine) tuple.
        Useful for advanced delegates (e.g. delegate multiple subprocesses in the same time).
        This is NOT a coroutine method.
        
        WARNING: this is not a safe way for asynchronous executing and get the result. Use `RoutineFuture` instead.
        
        :param subprocess: a coroutine
        
        :returns: (matcher, routine) where matcher is a event matcher to get the delegate result, routine is the created routine
        '''
        async def delegateroutine():
            try:
                r = await subprocess
            except:
                _, val, _ = sys.exc_info()
                e = RoutineControlEvent(RoutineControlEvent.DELEGATE_FINISHED, self.currentroutine,
                                        exception=val)
                self.scheduler.emergesend(e)
                raise
            else:
                e = RoutineControlEvent(RoutineControlEvent.DELEGATE_FINISHED, self.currentroutine,
                                        result = r)
                await self.wait_for_send(e)
        r = self.subroutine(generatorwrapper(delegateroutine(), 'subprocess', 'delegate'), True)
        finish = RoutineControlEvent.createMatcher(RoutineControlEvent.DELEGATE_FINISHED, r)
        return finish, r