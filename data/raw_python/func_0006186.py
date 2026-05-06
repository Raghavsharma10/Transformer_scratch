def resolve(self, value):
        """Resolves the promise with the given value."""
        if self is value:
            raise TypeError('Cannot resolve promise with itself.')

        if isinstance(value, Promise):
            value.done(self.resolve, self.reject)
            return

        if self._state != 'pending':
            raise RuntimeError('Promise is no longer pending.')

        self.value = value
        self._state = 'resolved'
        callbacks = self._callbacks
        self._callbacks = None
        for callback in callbacks:
            callback(value)