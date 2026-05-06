def reject(self, reason):
        """Rejects the promise with the given reason."""
        if self._state != 'pending':
            raise RuntimeError('Promise is no longer pending.')

        self.reason = reason
        self._state = 'rejected'
        errbacks = self._errbacks
        self._errbacks = None
        for errback in errbacks:
            errback(reason)