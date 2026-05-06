def start(self):
        """Schedule the timeout.  This is called on construction, so
        it should not be called explicitly, unless the timer has been
        canceled."""
        assert not self._timer, '%r is already started; to restart it, cancel it first' % self
        loop = evergreen.current.loop
        current = evergreen.current.task
        if self.seconds is None or self.seconds < 0:
            # "fake" timeout (never expires)
            self._timer = None
        elif self.exception is None or isinstance(self.exception, bool):
            # timeout that raises self
            self._timer = loop.call_later(self.seconds, self._timer_cb, current.throw, self)
        else:
            # regular timeout with user-provided exception
            self._timer = loop.call_later(self.seconds, self._timer_cb, current.throw, self.exception)