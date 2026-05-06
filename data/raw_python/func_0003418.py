async def wait(self, container = None):
        '''
        :param container: DEPRECATED container of current routine
        
        :return: The result, or raise the exception from set_exception.
        '''
        if hasattr(self, '_result'):
            if hasattr(self, '_exception'):
                raise self._exception
            else:
                return self._result
        else:
            ev = await FutureEvent.createMatcher(self)
            if hasattr(ev, 'exception'):
                raise ev.exception
            else:
                return ev.result