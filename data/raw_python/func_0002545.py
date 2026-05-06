def _add_error(self, message):
        """Add an error test to the suite."""
        error_line = Result(False, None, message, Directive(""))
        self._suite.addTest(Adapter(self._filename, error_line))