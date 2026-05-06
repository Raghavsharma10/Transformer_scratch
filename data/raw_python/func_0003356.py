async def end_delegate(self, delegate_matcher, routine = None, forceclose = False):
        """
        Retrieve a begin_delegate result. Must be called immediately after begin_delegate
        before any other `await`, or the result might be lost.
        
        Do not use this method without thinking. Always use `RoutineFuture` when possible.
        """
        try:
            ev = await delegate_matcher
            if hasattr(ev, 'exception'):
                raise ev.exception
            else:
                return ev.result
        finally:
            if forceclose and routine:
                routine.close()