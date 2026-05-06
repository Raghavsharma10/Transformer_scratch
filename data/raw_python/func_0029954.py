def logged_exception(self, e):
        """Record the exception, but don't log it; it's already been logged

        :param e:  Exception to log.

        """
        if str(e) not in self._errors:
            self._errors.append(str(e))

        self.set_error_state()
        self.buildstate.state.exception_type = str(e.__class__.__name__)
        self.buildstate.state.exception = str(e)