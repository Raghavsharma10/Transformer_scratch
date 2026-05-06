def addFailure(self, test, exception):
        """Register that a test ended with a failure.

        Parameters
        ----------
        test : unittest.TestCase
            The test that has completed.
        exception : tuple
            ``exc_info`` tuple ``(type, value, traceback)``.

        """
        result = self._handle_result(
            test, TestCompletionStatus.failure, exception=exception)
        self.failures.append(result)
        self._mirror_output = True