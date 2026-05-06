def disable_logger(self, disabled=True):
        """
        Disable all logging calls.
        """
        # Disable standard IO streams
        if disabled:
            sys.stdout = _original_stdout
            sys.stderr = _original_stderr
        else:
            sys.stdout = self.__stdout_stream
            sys.stderr = self.__stderr_stream

        # Disable handlers
        self.logger.disabled = disabled