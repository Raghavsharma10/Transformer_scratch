def set_exception(self, exception):
        '''
        Set an exception to Future object, wake up all the waiters
        
        :param exception: exception to set
        '''
        if hasattr(self, '_result'):
            raise ValueError('Cannot set the result twice')
        self._result = None
        self._exception = exception
        self._scheduler.emergesend(FutureEvent(self, exception = exception))