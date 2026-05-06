def done(self, on_success=None, on_failure=None):
        """Attaches some callbacks to the promise and returns the promise."""
        if on_success is not None:
            if self._state == 'pending':
                self._callbacks.append(on_success)
            elif self._state == 'resolved':
                on_success(self.value)
        if on_failure is not None:
            if self._state == 'pending':
                self._errbacks.append(on_failure)
            elif self._state == 'rejected':
                on_failure(self.reason)
        return self