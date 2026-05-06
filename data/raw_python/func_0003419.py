def set_result(self, result):
        '''
        Set the result to Future object, wake up all the waiters
        
        :param result: result to set
        '''
        if hasattr(self, '_result'):
            raise ValueError('Cannot set the result twice')
        self._result = result
        self._scheduler.emergesend(FutureEvent(self, result = result))