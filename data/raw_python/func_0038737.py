def kill(self, typ=TaskExit, value=None, tb=None):
        """Terminates the current task by raising an exception into it.
        Whatever that task might be doing; be it waiting for I/O or another
        primitive, it sees an exception as soon as it yields control.

        By default, this exception is TaskExit, but a specific exception
        may be specified.
        """
        if not self.is_alive():
            return
        if not value:
            value = typ()
        if not self._running:
            # task hasn't started yet and therefore throw won't work
            def just_raise():
                six.reraise(typ, value, tb)
            self.run = just_raise
            return
        evergreen.current.loop.call_soon(self.throw, typ, value, tb)