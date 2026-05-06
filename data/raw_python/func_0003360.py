async def execute_all_with_names(self, subprocesses, container = None, retnames = ('',), forceclose = True):
        '''
        DEPRECATED Execute all subprocesses and get the return values.
        
        :param subprocesses: sequence of subroutines (coroutines)
        
        :param container: if specified, run subprocesses in another container.
        
        :param retnames: DEPRECATED get return value from container.(name) for each name in retnames.
                         '' for return value (to be compatible with earlier versions)
        
        :param forceclose: force close the routines on exit, so all the subprocesses are terminated
                           on timeout if used with executeWithTimeout
        
        :returns: a list of tuples, one for each subprocess, with value of retnames inside:
                  `[('retvalue1',),('retvalue2',),...]`
        '''
        if not subprocesses:
            return []
        subprocesses = list(subprocesses)
        if len(subprocesses) == 1 and (container is None or container is self) and forceclose:
            # Directly run the process to improve performance
            return [await subprocesses[0]]
        if container is None:
            container = self
        delegates = [self.begin_delegate_other(p, container, retnames) for p in subprocesses]
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
                        container.terminate(d[1])
                    except Exception:
                        pass