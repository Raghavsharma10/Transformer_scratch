def intent(self, intent):
        ''' Decorator to register intent handler'''

        def _handler(func):
            self._handlers['IntentRequest'][intent] = func
            return func

        return _handler