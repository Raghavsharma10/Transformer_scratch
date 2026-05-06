def connect_cancel(self, functions):
        '''Run given functions when a run is cancelled.
        '''
        self._cancel_functions = []
        for func in functions:
            if isinstance(func, basestring) and hasattr(self, func) and callable(getattr(self, func)):
                self._cancel_functions.append(getattr(self, func))
            elif callable(func):
                self._cancel_functions.append(func)
            else:
                raise ValueError("Unknown function %s" % str(func))