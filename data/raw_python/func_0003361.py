async def execute_all(self, subprocesses, forceclose=True):
        '''
        Execute all subprocesses and get the return values.
        
        :param subprocesses: sequence of subroutines (coroutines)
        
        :param forceclose: force close the routines on exit, so all the subprocesses are terminated
                           on timeout if used with executeWithTimeout
        
        :returns: a list of return values for each subprocess
        '''
        if not subprocesses:
            return []
        subprocesses = list(subprocesses)
        if len(subprocesses) == 1 and forceclose:
            return [await subprocesses[0]]
        delegates = [self.begin_delegate(p) for p in subprocesses]
        matchers = [d[0] for d in delegates]
        try:
            _, eventdict = await self.wait_for_all(*matchers)
            events = [eventdict[m] for m in matchers]
            exceptions = [e.exception for e in events if hasattr(e, 'exception')]
            if exceptions:
                if len(exceptions) == 1:
                    raise exceptions[0]
                else:
                    raise MultipleException(exceptions)
            return [e.result for e in events]
        finally:
            if forceclose:
                for d in delegates:
                    try:
                        d[1].close()
                    except Exception:
                        pass