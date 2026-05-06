def _setup_stdout(self):
        """Hook stdout and stderr if buffering is enabled.

        """
        if self.buffer:
            if self._stderr_buffer is None:
                self._stderr_buffer = StringIO()
                self._stdout_buffer = StringIO()
            sys.stdout = self._stdout_buffer
            sys.stderr = self._stderr_buffer