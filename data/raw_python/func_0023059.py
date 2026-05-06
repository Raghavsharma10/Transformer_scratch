def request(self, request_type):
        ''' Decorator to register generic request handler '''

        def _handler(func):
            self._handlers[request_type] = func
            return func

        return _handler