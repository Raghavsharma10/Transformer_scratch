def redirect_stderr(self, enabled=True, log_level=logging.ERROR):
        """
        Redirect sys.stderr to file-like object.
        """
        if enabled:
            if self.__stderr_wrapper:
                self.__stderr_wrapper.update_log_level(log_level=log_level)
            else:
                self.__stderr_wrapper = StdErrWrapper(logger=self, log_level=log_level)

            self.__stderr_stream = self.__stderr_wrapper
        else:
            self.__stderr_stream = _original_stderr

        # Assign the new stream to sys.stderr
        sys.stderr = self.__stderr_stream