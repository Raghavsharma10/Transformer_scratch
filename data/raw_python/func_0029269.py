def addError(self, test, exception):
        """Register that a test ended in an error.

        Parameters
        ----------
        test : unittest.TestCase
            The test that has completed.
        exception : tuple
            ``exc_info`` tuple ``(type, value, traceback)``.

        """
        result = self._handle_result(
            test, TestCompletionStatus.error, exception=exception)
        self.errors.append(result)
        self._mirror_output = True