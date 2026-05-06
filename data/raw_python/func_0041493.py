def dead(self):
        """Whether the callback no longer exists.

        If the callback is maintained via a weak reference, and that
        weak reference has been collected, this will be true
        instead of false.
        """
        if not self._weak:
            return False
        cb = self._callback()
        if cb is None:
            return True
        return False