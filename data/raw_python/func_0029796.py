def error(self):
        """
        Return an instance of Exception if any, else None.

        Actually check for a :class:`TimeoutError` or a
        :class:`ExitCodeError`.
        """
        if self.__timed_out:
            return TimeoutError(self.session, self, "timeout")
        if self.__exit_code is not None and \
                self.__expected_exit_code is not None and \
                self.__exit_code != self.__expected_exit_code:
            return ExitCodeError(self.session, self,
                                 'bad exit code: Got %s' % self.__exit_code)